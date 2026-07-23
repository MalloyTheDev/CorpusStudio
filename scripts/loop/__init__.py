"""Bounded autonomous-loop CONTROLLER for CorpusStudio (the operational loop; Level 3 -> 4).

This package is the *doer's* control plane - the executable state machine that drives a goal from
intake to done-or-escalate. It is DISTINCT from the assurance/evidence plane (``scripts/assurance/``,
``cs_assure``): the assurance plane ANSWERS questions (what changed / obligations / gate / doc-trust);
this controller DECIDES what to do next and keeps the work bounded. The controller QUERIES the assurance
plane; it does not replace it.

In this environment the EXECUTOR is the LLM (or a delegated agent). The controller cannot run the
reasoning itself, so it is a deterministic scaffold: it emits the next action + its constraints, takes
back a classified observation, routes it, and transitions - so the loop advances without a human
prompting every step and STOPS or ESCALATES instead of retrying forever.
"""
