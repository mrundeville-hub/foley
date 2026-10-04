# Canonical terminal design reference

Source: `UI_REFERENCE.png`, copied unchanged from the user's original attachment.

The reference is a cobalt-blue live-code surface with white monospaced text, muted line numbers, inline note sequences, three compact punchcards, outlined active tokens and a thin vertical cursor. Graphs occupy roughly 80% of the available width. The lower graph is made of contiguous white slices separated by narrow blue lines.

2026-10-04 corrections:

- Material contacts carry the object's base resonance as continuous MIDI metadata. Equal-strength contacts with different resonances appear on different rows. Motion gestures carry the synthesis pitch contour and appear as stepped bends alongside the VOICE events.
- Inline note sequences use real score events. Active tokens are bracketed and underlined on a darker blue background, gated by the corresponding audio meter.
- Removed the full-cell inverse playhead; the cursor is a thin muted line.
- Removed misleading full-width bed lines from the pitched-voice graph.
- Slice fills use background color to avoid horizontal glyph seams; output amplitude remains available even when there are no explicit CHOP events.
- New video / Process / Replay / Vary / Files remain visible and work at 50×18, 80×24 and larger terminal sizes.

The terminal adapts the reference's composition rather than reproducing its screenshot pixel-for-pixel. The host terminal chooses the font, and the displayed score belongs to the current video rather than the example composition. The source reference is a still image, so its animation timing cannot be verified from that image.

Current visual evidence: `qa/material-engine/design.svg.png` (110×38) and `design-small.svg.png` (80×24), captured from the Textual app with a real material-engine render and a controlled preview clock. The native host terminal's font is not verified by this screenshot. At compact sizes, fewer chart rows fit. Workflow checks cover processing a different second video, preserving the first result, keyboard / mouse controls, player synchronization, resonance placement and gesture contours.
