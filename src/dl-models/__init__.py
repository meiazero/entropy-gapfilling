"""Deep learning models for satellite image gap-filling.

Each model lives in its own subpackage (ae/, vae/, gan/, transformer/)
and can be trained, evaluated, and used for inference independently.

Shared infrastructure (base class, dataset adapter, losses) is in shared/.
"""
