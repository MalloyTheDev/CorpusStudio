"""Reporting helpers that summarize a dataset project for humans.

These helpers only read existing project artifacts (project metadata, schema,
examples, splits, quality reports, evaluation reports) and render inspectable
summaries. They never launch training, publish datasets, or bypass validation.
"""
