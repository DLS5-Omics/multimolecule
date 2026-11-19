"""Model interpretability tools for MultiMolecule.

Use ``activation.capture`` for layer outputs and ``attention.capture`` for model-returned
attention weights. Use ``intervention.compare`` to compare predictions after activation
replacements, and ``jacobian.full``, ``jacobian.vjp``, or ``jacobian.jvp`` for derivatives.
``sae.SparseAutoencoder`` learns sparse representations of captured activations.
The algorithms in ``attribution`` use Captum through the optional ``interpret`` extra.
"""
