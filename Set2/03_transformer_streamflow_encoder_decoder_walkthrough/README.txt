Transformer demo package
========================

Files
-----
- index.html
- styles.css
- app.js

How to use
----------
Option 1: Double-click index.html
- Because this demo uses only local HTML/CSS/JS with no external dependencies,
  you can open index.html directly in a browser.

Option 2: Serve locally from a terminal
- macOS / Linux:
    cd transformer_streamflow_encoder_decoder_walkthrough
    python3 -m http.server 8000
- Then open:
    http://localhost:8000

Teaching notes
--------------
- Use the phase slider or Prev/Next buttons to walk through the architecture.
- Switch between Training mode and Inference mode.
- Change the decoder step to see how masked self-attention and cross-attention change.
