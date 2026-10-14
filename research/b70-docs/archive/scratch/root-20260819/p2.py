import os
filepath = '~/B70-DOCS/scripts/tmp/b70-qwen38-games-harness-vision.py'

with open(filepath, 'a', encoding='utf-8') as f:
    # 05-fluid-particles
    f.write('GAMES["05-fluid-particles"] = """')
    f.write('''Domain: Generative art / particle system piece.

MUST-HAVE FEATURES
1. WebGL 2.0 GPGPU particle system handling exactly 65,536 particles (stored in a 256x256 texture).
2. Ping-pong Framebuffer Object (FBO) technique: use fragment shaders to compute physics updates (positions and velocities) purely on the GPU.
3. Velocity field: Procedural 3D curl noise generated in the velocity update fragment shader, displacing the particles dynamically.
4. Color mapping: Particle colors must not be uniform; they must be derived from their instantaneous velocity magnitude (e.g., slow = dark blue, fast = bright cyan/white).
5. Fading trails: Implement a feedback loop / accumulation buffer that slightly dims the screen (e.g., 0.95 opacity multiplier) every frame, creating fluid-like motion trails.
6. Mouse interaction: A repulsive/attractive force applied to the velocity texture when the mouse is down, allowing users to part the fluid.
7. Auto-wander attractor: To ensure headless validation succeeds, implement an autonomous attractor point that wanders the screen in a complex Lissajous pattern, constantly stirring the particles without user input.
8. Bloom post-processing: Extract pixels above a brightness threshold, blur them using a two-pass Gaussian blur (horizontal/vertical), and add them back to the scene to make the particles glow.
9. Wrap-around bounds: Particles leaving the screen edge must smoothly teleport to the opposite side while maintaining their velocity, or bounce elastically.
10. Point rendering: Render the particles using `gl.POINTS`, scaling the point size dynamically based on the particle's lifespan or velocity (between 1.0 and 3.0 pixels).
11. Lifespan system: Particles should have a simulated lifespan. Once depleted, they respawn at a random location on the screen to maintain a steady density.
12. UI Controls: A minimal absolute-positioned HTML slider in the corner controlling the curl noise frequency (scale).
13. Responsive canvas: The WebGL viewport and coordinate system must scale perfectly with browser window resizing, mapping the texture coordinates to the new aspect ratio.
14. Blending setup: Use additive blending (`gl.blendFunc(gl.SRC_ALPHA, gl.ONE)`) for the particle rendering pass to create intense, overlapping light clusters.
15. Initial state constraint: Particles must spawn in a dense, mathematical formation (like a sphere or a grid) in the center of the screen before being blown apart by the curl noise.

ARCHITECTURE / IMPLEMENTATION NOTES
You must manage multiple WebGLPrograms. One for the physics update (GPGPU), one for rendering the points, one for the trail feedback, and one for the bloom post-processing.
Data textures (floating-point textures `gl.RGBA32F`) are mandatory for storing high-precision positions and velocities. Ensure the WebGL extension `EXT_color_buffer_float` is enabled.
Instead of uploading vertex data every frame, upload a static grid of UV coordinates (0.0 to 1.0) corresponding to the 256x256 texture space. The vertex shader for the render pass will use these UVs to sample the position texture and determine `gl_Position`.

POLISH
Visual fidelity: The curl noise must evolve over time (`u_time`) so the fluid currents constantly shift and never repeat a static pattern.
Color interpolation: Use a `mix()` function in GLSL to transition between a base color palette and a peak velocity color palette.
Framerate stability: Decouple the GPGPU physics steps from the frame rate if possible, or lock the delta time to prevent simulation explosion.

TECHNICAL CONSTRAINTS
Strictly a single HTML file. No external CSS, JS, or libraries (No Three.js, no TWGL).
Pure WebGL 2.0 API. Must run entirely offline.
Boot constraint: The simulation must start immediately.
Screenshot Expectation at t=3s: The initial grid of 65,536 particles must have been blown apart by the curl noise into swirling, organic, fluid-like tendrils. Additive blending and bloom should make the high-density areas glow brilliantly. The auto-wander attractor should be actively disrupting the field, proving the GPGPU physics step is working perfectly.''')
    f.write('"""\n\n')

    # 06-webaudio-visualizer
    f.write('GAMES["06-webaudio-visualizer"] = """')
    f.write('''Domain: Music visualizer with WebAudio.

MUST-HAVE FEATURES
1. Pure WebAudio API synthesizer. No `<audio>` tags, no MP3s, no external assets. All sound is mathematically synthesized in real-time.
2. Generative Sequencer: A JavaScript loop that procedurally plays a chord progression (e.g., Fm, C#M, G#M, D#M) using broken arpeggios at 120 BPM.
3. Synth architecture: At least two oscillator types. A bass drone (sine/triangle) and a melody arp (sawtooth/square).
4. Audio routing: Oscillators -> Envelope Generators (ADSR for volume) -> BiquadFilterNode (lowpass) -> Master Gain.
5. AnalyserNode integration: Hook the master output into an `AnalyserNode` with an FFT size of 1024 to extract real-time frequency data and waveform data.
6. Auto-play workaround: Modern browsers block autoplaying audio. Implement a massive "CLICK ANYWHERE TO START" overlay. However, the visualizer MUST run headlessly at t=0 by driving the visuals with a fake/simulated procedural sine wave until actual audio is unlocked.
7. 3D Visualizer (Canvas2D or WebGL): A wireframe tunnel or a geometric mandala that scales and twists based on the audio data.
8. Frequency mapping: Map the low-frequency (bass) FFT bins to the overall scale/zoom of the visualizer.
9. Treble mapping: Map the high-frequency FFT bins to the color intensity (e.g., shifting from blue to bright white).
10. Waveform mapping: Map the time-domain waveform data to the displacement of the vertices or lines in the tunnel/mandala, making them jagged when the saw wave plays.
11. Beat detection: A simple algorithm observing volume spikes in the bass bins to trigger sudden camera rotations or color palette shifts on the downbeat.
12. Particle emission: Spawn small floating geometric particles from the center that travel outward; their speed should multiply by the overall volume level.
13. Delay/Echo effect: Add a `DelayNode` to the WebAudio graph for the arpeggiator to create a rich, spatial soundscape.
14. Master Volume UI: An HTML range slider controlling the master gain.
15. Post-processing: Add a subtle chromatic aberration effect to the visuals that scales with the intensity of the audio volume.

ARCHITECTURE / IMPLEMENTATION NOTES
The WebAudio Context should be initialized only upon the first user interaction, but the render loop must start immediately.
Create a `SynthVoice` class that handles the instantiation of an `OscillatorNode`, applies an ADSR curve using `gainNode.gain.setValueAtTime` and `linearRampToValueAtTime`, and handles cleanup.
The sequencer should use `audioCtx.currentTime` to schedule notes perfectly in advance, rather than relying on `setInterval` which is notoriously inaccurate for audio.

POLISH
Visual aesthetics: The rendering should feel like an abstract demoscene entry. High contrast, glowing lines, deep black background.
Audio mixing: Ensure the frequencies don't clash. Keep the saw waves filtered so they aren't piercing, and ensure the bass is deep but not clipping (use a `DynamicsCompressorNode` on the master output).
Smooth transitions: Use `lerp` on the visual properties driven by the FFT data so the geometry doesn't jitter aggressively, but smoothly pulses.

TECHNICAL CONSTRAINTS
Strictly a single HTML file. No external dependencies.
Canvas2D or WebGL for visuals. Web Audio API for sound.
Screenshot Expectation at t=3s: The "CLICK TO START" overlay must be visible, BUT the background visualizer must be fully rendering its geometry in an idle, simulated state. It must show depth, geometric complexity, and the color palette clearly, proving the render loop operates independently of the audio lock.''')
    f.write('"""\n\n')

    # 07-physics-sandbox
    f.write('GAMES["07-physics-sandbox"] = """')
    f.write('''Domain: Physics sandbox.

MUST-HAVE FEATURES
1. Custom 2D physics engine implemented from scratch using Verlet integration for stability.
2. Rigid body definitions: Support for both Circles and Convex Polygons (rectangles, triangles).
3. Distance Constraints (Sticks): Mathematical links between two points that enforce a specific distance, allowing the creation of bridges, ropes, and ragdolls.
4. SAT (Separating Axis Theorem): Implement SAT for accurate collision detection between polygons, calculating the penetration depth and normal vector.
5. Collision Resolution: Resolve overlapping bodies by displacing them along the collision normal based on their relative mass, and applying appropriate bounce (restitution).
6. Friction: Apply velocity dampening parallel to the collision normal to simulate friction so stacked boxes don't slide endlessly.
7. Spatial Hashing / Grid Broad-phase: Divide the screen into a grid to quickly cull collision checks (O(N) complexity) instead of checking every object against every other object (O(N^2)).
8. Interactive Mouse Picker: Clicking and dragging applies a spring constraint to the selected object, pulling it towards the mouse cursor.
9. Gravity toggle: A UI button to flip gravity on, off, or reverse it.
10. Auto-spawn scenario: Upon load (for headless evaluation), automatically spawn a sturdy pyramid of 15 rectangular boxes, and drop a heavy circle onto the peak at t=1.0s.
11. Boundary walls: Static, unmovable line segments or heavy polygons acting as the floor, ceiling, and walls.
12. Spawner UI: Buttons to spawn new boxes, circles, and linked ragdolls at the top of the screen.
13. Debug rendering: Render the objects cleanly, but also draw faint lines indicating the constraints/sticks and the centers of mass.
14. Resting contact stability: The engine must put objects to "sleep" or aggressively dampen them when velocity is near zero to prevent jiggling stacks.
15. Color coding: Dynamic objects are colored by their mass/type, static walls are dark gray, and constraints are thin white lines.

ARCHITECTURE / IMPLEMENTATION NOTES
Structure the engine with distinct steps in the loop: 1. Apply forces (gravity). 2. Update positions (Verlet integration: `pos = pos + (pos - old_pos) + acc * dt^2`). 3. Satisfy constraints (multiple iterations, e.g., 5-10 loops of resolving sticks). 4. Broad-phase collision. 5. Narrow-phase collision (SAT). 6. Collision resolution.
Represent polygons as an array of local vertices and a transform (position, rotation).
Do not use external physics libraries like Matter.js or Box2D. You must write the math entirely inline.

POLISH
Visuals: Render the boxes with a slightly rounded corner effect if using Canvas2D (using `arcTo`), or a clean drop shadow (`ctx.shadowBlur`).
Interactivity: Add a particle burst effect when high-velocity collisions occur (e.g., when the heavy circle hits the pyramid).
Performance: Maintain a strict 60fps even with 300+ objects on screen by ensuring the spatial hashing works correctly.

TECHNICAL CONSTRAINTS
Strictly a single HTML file. Pure Canvas2D API or WebGL. No external libraries.
Offline, inline CSS/JS.
Screenshot Expectation at t=3s: The canvas must show the auto-spawned pyramid of boxes mid-collapse or reacting to the heavy circle impact. The physics solver must demonstrate non-penetrating rigid bodies. The UI spawner buttons must be visible at the top. The scene must prove the custom SAT and constraint math is functioning.''')
    f.write('"""\n\n')

    # 08-cyberpunk-terminal
    f.write('GAMES["08-cyberpunk-terminal"] = """')
    f.write('''Domain: Text/typing game or interactive fiction.

MUST-HAVE FEATURES
1. Pure HTML DOM interface simulating an old-school, glowing CRT terminal.
2. Typing effect system: All system output must render character-by-character with a configurable delay (e.g., 10ms-30ms) to simulate baud rates.
3. Virtual Filesystem (VFS): A JSON or object-based tree representing directories and text files (e.g., `/home/guest`, `/sys/logs/classified.doc`).
4. Command Parser: A text input field (`<input type="text">`) that accepts commands: `help`, `ls`, `cd`, `cat`, `clear`, `whoami`, `sudo`, and `exploit`.
5. Error handling: Typing invalid commands yields "Command not found", and trying to `cd` into a file yields "Not a directory".
6. Hacking minigame (`exploit` command): Triggering this drops the user into a timed minigame where they must accurately type a 30-character sequence of random hex codes within 15 seconds.
7. Auto-boot sequence: To ensure headless validation, the terminal must automatically power on, display an ASCII art banner, and simulate typing `sys_check` and `cd /root` over the first 3 seconds without user input.
8. Permission system: Certain directories require the user to complete the `exploit` minigame to gain "root" privileges.
9. CRT Visuals: Apply CSS `text-shadow` for glow, a repeating linear-gradient for scanlines, an animation for screen flicker, and `border-radius` with a subtle box-shadow inset to curve the screen.
10. Persistent history: Pressing the Up/Down arrows recalls previously entered commands.
11. Scroll management: The terminal output div must automatically scroll to the bottom whenever a new character is appended.
12. Blinking cursor: A CSS-animated block (`█`) sitting at the end of the current input line.
13. Audio simulated visually: Since no audio is required, flash the screen slightly red when an error occurs, or green when access is granted.
14. Progress bars: Certain simulated commands (like `connect`) should output ASCII progress bars (`[#####.....] 50%`) that update on the same line.
15. Story payload: Reading specific files via `cat` reveals a cyberpunk narrative about an rogue AI escaping containment.

ARCHITECTURE / IMPLEMENTATION NOTES
Do not use Canvas or WebGL. This must be an exercise in pure DOM manipulation, CSS, and asynchronous JavaScript.
Use `async/await` and a `sleep(ms)` utility function heavily to manage the timing of the text output sequences.
Maintain a state object tracking `currentPath`, `userRole`, and `commandHistory`.
The input field should be invisible; capture global keystrokes and map them to a visible DOM span to ensure full control over the cursor and formatting.

POLISH
Aesthetic: The background should be deep black (`#050505`), the text should be terminal green (`#0f0`) or amber (`#ffb000`).
Typographic detail: Use a monospace font strictly. Align columns perfectly when running `ls -l`.
Scanline rendering: The scanlines must slowly drift downwards using a CSS `@keyframes` animation on `background-position`.

TECHNICAL CONSTRAINTS
Strictly a single HTML file. Inline CSS/JS.
Must work offline without fetching external fonts (use `font-family: monospace, "Courier New"`).
Screenshot Expectation at t=3s: The auto-boot sequence must be mid-execution or completed. The screen must show the ASCII art banner, the simulated typed commands, and the glowing CRT styling with scanlines. The terminal must not be empty.''')
    f.write('"""\n\n')

