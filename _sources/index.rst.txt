Welcome to Yoke's Documentation
===============================

.. warning::

   🚧 **This section is under active construction!**  
   Expect rough edges, placeholder content, and ongoing changes.

**Yoke** is a modular, flexible deep learning library designed for advanced vision tasks,
particularly those involving token-based architectures such as Vision Transformers (ViTs)
and SWIN Transformers. Yoke provides building blocks for patch embedding, hierarchical
attention, loss computation, training loops, and more.

Whether you're prototyping research or building production-grade systems, Yoke helps you
get from raw image data to well-structured transformer pipelines with minimal friction.

📦 **Key Features**
-------------------
- Patch-based image embedding with SWIN-style token merging
- Modular transformer blocks with clean attention mechanisms
- Advanced loss functions for vision tasks
- Lightweight training utilities and logging
- Full support for custom dataset integration

📚 **Getting Started**
----------------------
To begin using Yoke, check out the :doc:`modules` section for an overview of all
available classes, functions, and utilities.

If you're looking to extend or contribute, the code is structured to be highly
navigable and follows consistent design principles.

**Documentation Contents**
--------------------------
.. toctree::
   :maxdepth: 2
   :caption: Concepts

   harnesses
   start_study
   loderunner_animate
   prediction_modes

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   modules

📖 About This Documentation
---------------------------
This documentation is auto-generated using Sphinx and includes full docstrings for all
functions, classes, and modules. For math-heavy operations (like patch reshaping or
transformer attention), formulas are rendered using LaTeX via MathJax.
