"""
Triage ReAct Agent — KB-Driven Reasoning with Structured Logging
==================================================================
Implements a ReAct (Reasoning and Acting) agent for ATC-20 structural
triage classification. Uses a local LLM (Ollama/vLLM) with three tools:
  - retrieve_text: Semantic search on ATC-20 knowledge base
  - verify: Evidence verification via sub-agent + KB grounding
  - refuse: Explicit refusal when evidence is insufficient

Key improvements over the original:
  1. KB-driven evidence matching (no hardcoded phrases)
  2. Structured JSON logging with full provenance
  3. Type-safe TriageResult output
  4. Configurable via centralized PipelineConfig

References:
  - Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)
  - Lewis et al., "Retrieval-Augmented Generation" (NeurIPS 2020)
"""

import datetime
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from openai import OpenAI

from config import PipelineConfig, TriageResult
from unified_retrieval import UnifiedKnowledgeBase


# ─────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger("triage_agent")


class TriageReActAgent:
    """ReAct agent for structural damage triage classification.
    
    Uses a local LLM via OpenAI-compatible API to classify buildings
    into ATC-20 placard categories (Green/Yellow/Red) based on
    point-cloud-derived geometric features and retrieved KB criteria.
    
    Args:
        kb_interface: UnifiedKnowledgeBase instance.
        config: PipelineConfig instance.
    """

    def __init__(self, kb_interface: UnifiedKnowledgeBase,
                 perception_model=None,
                 config: PipelineConfig = None,
                 api_base: str = None,
                 model_name: str = None):
        self.cfg = config or PipelineConfig()
        self.kb = kb_interface
        self.perception = perception_model
        self.client = OpenAI(
            base_url=api_base or self.cfg.api_base,
            api_key=self.cfg.api_key,
        )
        self.model_name = model_name or self.cfg.model_name
        self.max_steps = self.cfg.max_react_steps
        self.kb_corpus = self.kb.get_corpus_text()

        self.system_prompt = """You are a verification-driven reasoning agent for post-earthquake building triage.
Your job is to classify the damage state of a building (Green, Yellow, or Red placard) using point-cloud-derived numeric payloads and retrieved criteria.

Do not rely on free-text descriptions unless no point-cloud payload is available.

You strictly output in this format:
Thought: <what you are thinking>
Action: <one of: retrieve_text, verify, refuse>
Action Input: <the query string (for retrieve_text), or JSON mapping for verify/refuse>

When verified confidently, provide the final answer:
ANSWER: {"class": "Yellow", "citations": ["doc1", "doc2"], "confidence": 0.85, "reasoning_trace": ["thought1"]}

Available Actions:
- retrieve_text: Input a specific, concrete semantic search query like "Red Placard structural criteria" or "Yellow Placard ATC-20 rules" to search manuals. DO NOT input long conversational queries or questions, just the keywords needed to find the criteria.
- verify: Input {"class": "Red/Yellow/Green", "evidence": ["<Copy exactly the structural criteria from your retrieved text that justifies this classification based on the point-cloud evidence>"]} to run the verification sub-agent.
- refuse: Input {"reason": "<reason for refusal>"} explicitly when the retrieved text does not strongly support a definitive classification or the point-cloud payload is too ambiguous. Use this to prevent hallucination.
"""

    def _call_llm(self, messages: list) -> str:
        """Call the local LLM with the given message history."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.cfg.llm_temperature,
            max_tokens=self.cfg.llm_max_tokens,
            stop=["Observation:"],
        )
        return response.choices[0].message.content.strip()

    # ─────────────────────────────────────────────────────────
    # KB-Driven Evidence Verification (replaces hardcoded matching)
    # ─────────────────────────────────────────────────────────

    def _kb_evidence_supports(self, damage_class: str, evidence: list) -> bool:
        """Check if evidence supports the given damage class using KB retrieval.
        
        Instead of hardcoded phrase matching, this method:
        1. Retrieves the top-k KB passages for the damage class
        2. Computes token overlap between evidence and KB passages
        3. Returns True if sufficient overlap is found
        
        This is fully data-driven: updating kb_documents/ automatically
        updates the classification rules.
        """
        query = f"{damage_class} Placard ATC-20 structural criteria"
        kb_hits = self.kb.retrieve_text(query, k=self.cfg.kb_retrieval_top_k)

        if not kb_hits:
            return False

        # Combine all KB text for this class
        kb_text = " ".join(h.get("text", "").lower() for h in kb_hits)
        kb_tokens = set(re.findall(r"\w+", kb_text))

        # Combine all evidence text
        ev_text = " ".join(str(e).lower() for e in evidence)
        ev_tokens = set(re.findall(r"\w+", ev_text))

        if not ev_tokens:
            return False

        # Token overlap ratio
        overlap = len(ev_tokens & kb_tokens) / max(1, len(ev_tokens))
        return overlap >= self.cfg.evidence_overlap_threshold

    def verifier_sub_agent(self, damage_class: str, evidence: list) -> str:
        """Verification sub-agent — validates evidence against KB criteria.
        
        Uses KB-driven matching (no hardcoded phrases) to check if the
        evidence supports the claimed damage class.
        """
        # First: KB-driven check (fast, deterministic)
        if self._kb_evidence_supports(damage_class, evidence):
            return "VERIFIED"

        # Second: LLM-based verification (slower, more nuanced)
        prompt = (
            f"Verify if this evidence supports a {damage_class} damage classification. "
            f"Accept absence-based Green evidence if it directly matches KB criteria "
            f"and is not a contradiction. "
            f"Evidence: {evidence}\nReply ONLY with 'VERIFIED' or 'NEEDS_REVISION'."
        )
        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            ).choices[0].message.content.strip()
            return res
        except Exception as e:
            logger.warning(f"Verifier LLM call failed: {e}")
            return "NEEDS_REVISION"

    def _evidence_sanitizer(self, evidence_list: list,
                            damage_class: str = None) -> Tuple[bool, object]:
        """Sanitize evidence before verification.
        
        Returns (ok: bool, cleaned_evidence: list | reason: str).
        Rejects evidence that is empty or not grounded in the KB corpus.
        """
        cleaned = []
        corpus = self.kb_corpus or ""
        corpus_tokens = set(re.findall(r"\w+", corpus)) if corpus else set()

        # Class-specific keywords for heuristic matching
        class_keywords = {
            "red": ["collapse", "pancak", "sever", "lean", "out-of-plumb",
                     "failure", "shear", "displacement", "pancaking"],
            "yellow": ["yield", "crack", "cracking", "spalling", "parapet",
                       "deform", "damage", "yielding", "moderate"],
            "green": ["no apparent", "no significant", "minor", "intact",
                       "no restriction", "no damage"],
        }

        for ev in evidence_list:
            ev_text = str(ev).strip()
            ev_low = " ".join(ev_text.lower().split())
            if not ev_low:
                return (False, "Empty evidence item")

            # 1. Direct substring match in corpus
            if corpus and ev_low in corpus:
                cleaned.append(ev_text)
                continue

            # 2. Clause-level matching
            clauses = [c.strip() for c in re.split(r"[.;]\s*", ev_low) if c.strip()]
            clause_match = any(len(c) > 20 and c in corpus for c in clauses)
            if clause_match:
                cleaned.append(ev_text)
                continue

            # 3. Token overlap
            ev_tokens = set(re.findall(r"\w+", ev_low))
            if not ev_tokens:
                return (False, "Evidence contains no tokens")
            overlap = len(ev_tokens & corpus_tokens) / max(1, len(ev_tokens))
            if overlap >= self.cfg.evidence_overlap_threshold:
                cleaned.append(ev_text)
                continue

            # 4. Class keyword heuristic
            kw_matches = 0
            if damage_class and damage_class.lower() in class_keywords:
                kws = class_keywords[damage_class.lower()]
            else:
                kws = sum(class_keywords.values(), [])
            for kw in kws:
                if kw in ev_low:
                    kw_matches += 1
            if kw_matches >= 1 and overlap >= self.cfg.evidence_keyword_overlap_threshold:
                cleaned.append(ev_text)
                continue

            # 5. KB retrieval fallback
            try:
                hits = self.kb.retrieve_text(ev_low, k=3)
                for h in hits:
                    h_text = h.get("text", "").lower()
                    h_tokens = set(re.findall(r"\w+", h_text))
                    if len(ev_tokens & h_tokens) / max(1, len(ev_tokens)) >= 0.4:
                        cleaned.append(ev_text)
                        clause_match = True
                        break
                if clause_match:
                    continue
            except Exception:
                pass

            return (False, f"Evidence not grounded in KB (overlap {overlap:.2f}): {ev_text[:120]}")

        if not cleaned:
            return (False, "No evidence provided")
        return (True, cleaned)

    # ─────────────────────────────────────────────────────────
    # Agent Output Parser
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def parse_answer(raw_output: str) -> TriageResult:
        """Parse the LLM's ANSWER into a structured TriageResult.
        
        Handles both well-formed JSON and malformed outputs gracefully.
        """
        result = TriageResult(raw_output=raw_output)

        # Try to extract JSON from ANSWER: line
        answer_match = re.search(r"ANSWER:\s*(\{.*\})", raw_output, re.DOTALL)
        if answer_match:
            json_str = answer_match.group(1).strip()
            # Strip markdown code blocks if present
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()
            
            try:
                data = json.loads(json_str)
                result.predicted_class = data.get("class", "INCONCLUSIVE")
                result.confidence = float(data.get("confidence", 0.0))
                result.citations = data.get("citations", [])
                result.reasoning_trace = data.get("reasoning_trace", [])
                return result
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: try to extract class from text
        for cls in ["Green", "Yellow", "Red"]:
            if cls.lower() in raw_output.lower():
                result.predicted_class = cls
                result.confidence = 0.3  # Low confidence for fallback
                break

        return result

    # ─────────────────────────────────────────────────────────
    # Main ReAct Loop
    # ─────────────────────────────────────────────────────────

    def run(self, description: str = None,
            point_cloud_payload: dict = None,
            log_dir: str = None) -> Tuple[str, str]:
        """Execute the ReAct triage loop.
        
        Args:
            description: Natural language description of the building.
            point_cloud_payload: Structured numeric feature payload.
            log_dir: Directory for structured logs.
        
        Returns:
            (classification_string, trace_summary) tuple.
            Also stores self.last_result (TriageResult) for programmatic access.
        """
        t_start = time.time()
        print("\n[START] Starting Agentic Triage Loop...")

        # Build input content
        payload_text = None
        if point_cloud_payload is not None:
            try:
                payload_text = json.dumps(point_cloud_payload, indent=2, sort_keys=True)
            except Exception:
                payload_text = str(point_cloud_payload)

        run_source = payload_text if payload_text is not None else str(description or "")
        if payload_text is not None:
            user_content = (
                "New building data received. Point-cloud payload: "
                f"{payload_text}. Begin ReAct triage sequence using only "
                "the numeric payload and retrieved criteria."
            )
        else:
            user_content = (
                f"New building data received. Description: {description}. "
                "Begin ReAct triage sequence by searching for text criteria."
            )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Prepare structured log
        base_log_dir = Path(log_dir or self.cfg.get_logs_dir())
        base_log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_hash = hashlib.sha1(run_source.encode("utf-8")).hexdigest()[:8]
        log_path = base_log_dir / f"triage_{ts}_{run_hash}.json"

        structured_log = {
            "timestamp": ts,
            "run_hash": run_hash,
            "model_name": self.model_name,
            "input_type": "payload" if payload_text else "description",
            "config": {
                "max_steps": self.max_steps,
                "temperature": self.cfg.llm_temperature,
                "evidence_threshold": self.cfg.evidence_overlap_threshold,
            },
            "steps": [],
            "result": None,
        }

        last_observation = None
        repeat_count = 0
        total_tokens = 0

        for step in range(self.max_steps):
            step_log = {"step": step + 1, "action": None, "observation": None}
            print(f"\n--- Step {step + 1} ---")

            try:
                response = self._call_llm(messages)
            except Exception as e:
                print(f"[ERR] LLM call failed: {e}")
                step_log["error"] = str(e)
                structured_log["steps"].append(step_log)
                break

            print(f"[LLM] LLM:\n{response}")
            step_log["llm_response"] = response
            messages.append({"role": "assistant", "content": response})

            # Check for final answer
            if "ANSWER:" in response:
                print("\n[DONE] FINAL OUTPUT REACHED.")
                result = self.parse_answer(response)
                result.num_steps = step + 1
                result.latency_seconds = time.time() - t_start
                result.log_path = str(log_path)
                self.last_result = result

                structured_log["result"] = result.to_dict()
                structured_log["steps"].append(step_log)
                self._save_log(structured_log, log_path)

                return response, "Trace Complete"

            # Parse action
            if "Action:" in response:
                try:
                    action_line = [l for l in response.split("\n") if l.startswith("Action:")][0]
                    action = action_line.replace("Action:", "").strip()
                except Exception:
                    action = ""
                try:
                    input_line = [l for l in response.split("\n") if l.startswith("Action Input:")][0]
                    action_input = input_line.replace("Action Input:", "").strip()
                except Exception:
                    action_input = ""

                step_log["action"] = action
                step_log["action_input"] = action_input
                observation = ""

                if action == "retrieve_text":
                    res = self.kb.retrieve_text(action_input, k=self.cfg.kb_retrieval_top_k)
                    observation = " | ".join([f"{r['source']}: {r['text']}" for r in res])

                elif action == "verify":
                    try:
                        args = json.loads(action_input)
                        ok, cleaned = self._evidence_sanitizer(
                            args.get("evidence", []),
                            damage_class=args.get("class"),
                        )
                        if not ok:
                            observation = f"NEEDS_REVISION (Evidence sanitizer: {cleaned})"
                            step_log["sanitizer_result"] = {"ok": False, "reason": cleaned}
                        else:
                            observation = self.verifier_sub_agent(args["class"], cleaned)
                            step_log["sanitizer_result"] = {"ok": True, "cleaned_count": len(cleaned)}
                    except Exception:
                        observation = "NEEDS_REVISION (Malformed dict)"

                elif action == "refuse":
                    try:
                        args = json.loads(action_input)
                        reason = args.get("reason", "unspecified")
                        print(f"[REFUSED] REFUSED: {reason}")
                        result = TriageResult(
                            predicted_class="INCONCLUSIVE",
                            is_refused=True,
                            refuse_reason=reason,
                            num_steps=step + 1,
                            latency_seconds=time.time() - t_start,
                            log_path=str(log_path),
                        )
                        self.last_result = result
                        structured_log["result"] = result.to_dict()
                        structured_log["steps"].append(step_log)
                        self._save_log(structured_log, log_path)
                    except Exception:
                        pass
                    return "INCONCLUSIVE — defer to human reviewer.", "Trace Complete"
                else:
                    observation = "Unknown Action."

                print(f"[OBS] OBSERVATION: {observation}")
                step_log["observation"] = observation

                # Detect repeated NEEDS_REVISION cycles
                if isinstance(observation, str) and observation.strip().startswith("NEEDS_REVISION"):
                    if observation == last_observation:
                        repeat_count += 1
                    else:
                        repeat_count = 1
                    last_observation = observation

                    if repeat_count >= self.cfg.verifier_revision_patience:
                        # Auto-augment with more KB context
                        try:
                            parsed = None
                            try:
                                parsed = json.loads(action_input)
                            except Exception:
                                pass
                            query = (f"{parsed.get('class', '')} Placard ATC-20 criteria"
                                     if parsed and isinstance(parsed, dict) and parsed.get("class")
                                     else "ATC-20 criteria")
                            extra = self.kb.retrieve_text(query, k=self.cfg.kb_retrieval_top_k)
                            if extra:
                                extra_obs = "Additional KB passages: " + " | ".join(
                                    [f"{e['source']}: {e['text']}" for e in extra]
                                )
                                step_log["auto_augment"] = True
                                messages.append({"role": "user",
                                                "content": f"Observation: {observation}\n"
                                                           f"Auto-augment: {extra_obs}"})
                                repeat_count = 0
                                structured_log["steps"].append(step_log)
                                continue
                            else:
                                result = TriageResult(
                                    predicted_class="INCONCLUSIVE",
                                    num_steps=step + 1,
                                    latency_seconds=time.time() - t_start,
                                    log_path=str(log_path),
                                )
                                self.last_result = result
                                structured_log["result"] = result.to_dict()
                                structured_log["steps"].append(step_log)
                                self._save_log(structured_log, log_path)
                                return "INCONCLUSIVE — defer to human reviewer.", "Trace Complete"
                        except Exception:
                            result = TriageResult(
                                predicted_class="INCONCLUSIVE",
                                num_steps=step + 1,
                                latency_seconds=time.time() - t_start,
                                log_path=str(log_path),
                            )
                            self.last_result = result
                            structured_log["result"] = result.to_dict()
                            structured_log["steps"].append(step_log)
                            self._save_log(structured_log, log_path)
                            return "INCONCLUSIVE — defer to human reviewer.", "Trace Complete"
                else:
                    last_observation = observation
                    repeat_count = 0

                # Provide guidance for NEEDS_REVISION
                if isinstance(observation, str) and observation.strip().startswith("NEEDS_REVISION"):
                    guidance = (
                        "Verifier requested revision. Please retrieve additional KB passages "
                        "or refine the evidence. If the sanitized evidence directly matches "
                        "KB criteria, you may proceed to return the final ANSWER with citations. "
                        "If ambiguous, use the refuse action with a clear reason."
                    )
                    messages.append({"role": "user",
                                    "content": f"Observation: {observation}\nGuidance: {guidance}"})
                else:
                    messages.append({"role": "user", "content": f"Observation: {observation}"})

            structured_log["steps"].append(step_log)

        # Exceeded max steps - try to parse the last response as a fallback
        # instead of unconditionally returning INCONCLUSIVE
        result = self.parse_answer(response if 'response' in locals() else "INCONCLUSIVE")
        if result.predicted_class == "INCONCLUSIVE":
            result.predicted_class = "INCONCLUSIVE (Max Steps)"
        result.num_steps = self.max_steps
        result.latency_seconds = time.time() - t_start
        result.log_path = str(log_path)
        
        self.last_result = result
        structured_log["result"] = result.to_dict()
        self._save_log(structured_log, log_path)

        return response if 'response' in locals() else "INCONCLUSIVE", "Trace Complete"

    def _save_log(self, log_data: dict, path: Path):
        """Save structured log to JSON file."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Log saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save log: {e}")


if __name__ == "__main__":
    kb = UnifiedKnowledgeBase()
    agent = TriageReActAgent(kb_interface=kb)

    # Test the parser
    test_answer = 'ANSWER: {"class": "Yellow", "citations": ["ATC_20_Criteria.txt"], "confidence": 0.82, "reasoning_trace": ["observed moderate cracking"]}'
    result = TriageReActAgent.parse_answer(test_answer)
    print(f"Parsed class: {result.predicted_class}")
    print(f"Parsed confidence: {result.confidence}")
    print(f"Parsed citations: {result.citations}")
    assert result.predicted_class == "Yellow"
    print("[OK] Parser test passed!")
