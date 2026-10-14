import os

filepath = '~/B70-DOCS/scripts/tmp/b70-qwen38-games-harness-vision.py'
os.makedirs(os.path.dirname(filepath), exist_ok=True)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write('''# -*- coding: utf-8 -*-
"""
Qwen3.8-27B GPTQ vision-capable benchmark campaign harness.
Agent setup: pi agent, off/medium thinking, 2026-08-16.
10 ultra-detailed creative-coding prompts across domains (vision-verified campaign).
"""

GAMES = {}
''')

with open(filepath, 'a', encoding='utf-8') as f:
    # 01-temple-run
    f.write('GAMES["01-temple-run"] = """')
    f.write('''Domain: 3D endless runner.

MUST-HAVE FEATURES
1. Pure WebGL 2.0 implementation without Three.js, using raw shader compilation and buffer management.
2. Infinite procedural track generation: spawn track segments stretching from z=0 to z=-200.
3. Character controller: a primitive cube representing the player that auto-runs forward at a baseline speed of 20 units/sec.
4. Speed scaling: player speed increases by 2 units/sec every 10 seconds of survival, capping at 50 units/sec.
5. Lane system: 3 distinct lanes (-4, 0, 4 on the x-axis). Player switches lanes instantly with left/right arrows, with a 150ms cooldown.
6. Jumping mechanics: up arrow applies an upward impulse of 15 units/sec. Gravity is constantly -40 units/sec^2. Ground plane is at y=0.
7. Ducking mechanics: down arrow squashes the player cube's y-scale to 0.5 and doubles gravity to snap them to the ground quickly.
8. Obstacles: spawn hurdles (require jumping) and overhead beams (require ducking) every 30-50 units along the track.
9. Collision detection: AABB (Axis-Aligned Bounding Box) collision between the player and obstacles.
10. Coins: floating yellow octahedrons spaced every 10 units in random lanes, rotating at 2 rad/sec.
11. Score tracking: 10 points per coin, plus 1 point per 10 units traveled.
12. Camera setup: Follow camera locked to the player's x/z position, offset by y=10, z=15, looking slightly downward at an angle of 15 degrees.
13. Lighting: A directional light simulating the sun (diffuse shading), plus ambient illumination (vec3(0.2, 0.2, 0.3)).
14. Skybox/Background: a gradient background moving from deep dark blue at the top to horizon orange at the bottom, cleared every frame.
15. Game Over state: upon collision with an obstacle, instantly halt forward movement, turn the player mesh red, and display "GAME OVER".

ARCHITECTURE / IMPLEMENTATION NOTES
Implement a monolithic standard game loop using `requestAnimationFrame`. Use a finite state machine for the game states: `START`, `PLAYING`, `GAMEOVER`.
Structure your WebGL boilerplate cleanly: a function to compile shaders, a function to create buffer objects, and a central `render(deltaTime)` function.
Since you cannot use external matrix libraries like glMatrix, you must implement a minimal inline math library containing `mat4.perspective`, `mat4.lookAt`, `mat4.multiply`, `mat4.translate`, `mat4.scale`, and `mat4.rotate`. Ensure these mathematical implementations are highly optimized for a 60fps loop.
Manage memory carefully: do not create new buffers every frame. Instead, update a rolling array of track segment transformations and pass them as instanced data or uniform arrays, updating their z-positions as the player moves past them. Wrap the track segments back to the front to create the illusion of infinite generation without garbage collection pauses.

POLISH
HUD: An HTML overlay sitting absolute-positioned on top of the canvas, displaying the score in a large monospace font (24px, #FFF, text-shadow).
Particles: When a coin is collected, spawn a burst of 10 small yellow square particles that scatter outward and fade opacity over 400ms using a simple verlet integration step.
Shadow mapping: implement a basic depth map pass so the player and obstacles cast hard shadows onto the track plane, enhancing depth perception.
Accessibility/Controls: Pressing 'P' or 'Escape' pauses the `requestAnimationFrame` loop, dimming the screen via a CSS overlay and showing a "PAUSED" text. Ensure mouse clicks on the canvas also trigger jump/duck (left click jump, right click duck) for accessibility.

TECHNICAL CONSTRAINTS
Strictly a single self-contained HTML file. No external CSS, no external JS.
ZERO external libraries. No Three.js, no glMatrix, no GSAP. Pure DOM and WebGL 2.0 API.
Must function entirely offline without fetching external assets, textures, or fonts.
Boot constraint: The game must enter the `PLAYING` state automatically upon load without waiting for user interaction.
Auto-run constraint: To ensure headless screenshots capture meaningful gameplay, the character must auto-start running immediately. Obstacles must not appear in the first 60 units of the z-axis (giving roughly 3 seconds of clear running).
Screenshot Expectation at t=3s: The canvas must show the 3D track, the player cube moving forward, the score updating (greater than zero), lighting and shadows clearly active, and the first set of coins and obstacles appearing in the distance (z=-60 to z=-100). The scene must mathematically prove correct projection matrices, depth buffering, and backface culling.''')
    f.write('"""\n\n')

    # 02-webgl-raytracer
    f.write('GAMES["02-webgl-raytracer"] = """')
    f.write('''Domain: Real-time 3D renderer demo.

MUST-HAVE FEATURES
1. Pure WebGL 2.0 fragment shader raytracer covering a full-screen quad.
2. Ray generation mapping normalized device coordinates to a perspective camera frustum.
3. Analytic sphere intersection function (quadratic equation solver) supporting at least 4 distinct spheres.
4. Analytic plane intersection for an infinite floor plane at y=0.
5. Soft shadows using a penumbra estimation technique during shadow ray traversal (or area light sampling).
6. Recursive reflection loop (or unrolled loop for GLSL constraints) supporting up to 3 reflection bounces.
7. Multi-material system: Diffuse (matte), Specular (mirror reflection), and Dielectric (glass refraction with Snell's law).
8. Scene composition: One large diffuse sphere, one mirror sphere, one glass sphere, and a checkered floor pattern.
9. Dynamic point light source orbiting the scene (radius 15, height 10, speed 0.5 rad/sec).
10. Camera orbiting the center of the scene at a radius of 12 units, slowly rotating at 0.1 rad/sec.
11. Anti-aliasing achieved by casting 4 slightly jittered rays per pixel and averaging the results.
12. Ambient Occlusion (fake or ray-marched) to ground the objects on the floor plane.
13. Gamma correction (pow(color, 1.0/2.2)) applied at the very end of the fragment shader.
14. Vignette post-processing effect darkening the corners of the screen.
15. Interactive mouse look: clicking and dragging the canvas overrides the auto-orbit and manually rotates the camera.

ARCHITECTURE / IMPLEMENTATION NOTES
The JavaScript side should be absolute minimal boilerplate: compile a vertex shader that passes through a single large triangle or two quads, compile the complex fragment shader, and set up the `requestAnimationFrame` loop to pass uniform variables (`u_time`, `u_resolution`, `u_mouse`).
The fragment shader must contain the entire scene logic. Structure it with modular functions: `hitSphere`, `hitPlane`, `mapScene` (to find the closest intersection), `calcNormal`, `calcLighting`, and the main `renderRay` function.
For reflections and refractions, since true recursion is not supported in earlier GLSL versions, use an iterative loop in the fragment shader that updates the ray origin, ray direction, and attenuation mask per bounce.
The checkerboard pattern on the floor should be generated mathematically using the `mod` or `fract` functions on the world-space intersection coordinates, avoiding any external texture lookups.

POLISH
FPS Counter: Implement an HTML DOM overlay in the top left corner displaying real-time frames-per-second, updated every 500ms to evaluate shader performance.
Material properties: Ensure the glass material has a subtle chromatic aberration effect by splitting the refraction ray into red, green, and blue components with slightly different indices of refraction (e.g., IOR_R=1.50, IOR_G=1.52, IOR_B=1.54).
Lighting model: Use a physically-based Blinn-Phong or basic PBR approximation for the specular highlights so they look realistic and sharp on the mirror sphere but slightly blurred on the diffuse sphere.
Smooth Camera Interpolation: If the user interacts via mouse, smoothly interpolate (`lerp`) the camera position from the auto-orbit path to the manual override path to prevent jarring snapping.

TECHNICAL CONSTRAINTS
Strictly a single self-contained HTML file. Inline CSS for the canvas and FPS counter.
Must function offline. No network requests, no external images, no Three.js, no Shadertoy embeds. Pure WebGL API.
Resolution scaling: The canvas must scale to the window size, but internal render resolution should be clamped to 1920x1080 to ensure performance on lower-end GPUs during the 3-second evaluation window.
Boot constraint: The shader must compile and render immediately upon loading.
Screenshot Expectation at t=3s: The canvas must show a highly realistic, raytraced scene. The orbiting light source should be casting visible soft shadows on the checkerboard floor. The glass sphere must show background inversion (refraction), and the mirror sphere must show reflections of the other spheres. The camera should have auto-orbited slightly from the starting position, proving time uniform integration.''')
    f.write('"""\n\n')

    # 03-motion-poster
    f.write('GAMES["03-motion-poster"] = """')
    f.write('''Domain: Animation/motion-design piece.

MUST-HAVE FEATURES
1. HTML5 Canvas 2D implementation for a 60fps kinetic typography poster.
2. Large central typography reading "QWEN 3.8".
3. Font vectorization: Since external fonts are forbidden, programmatically draw the letters using geometric paths, or use `measureText` and `getImageData` on a hidden offscreen canvas to sample the pixel coordinates of the text.
4. Particle generation: Convert the sampled text pixels into a grid of 2,000 to 5,000 distinct particle points.
5. Spring physics solver (Hooke's law): Every particle must have an origin position, current position, velocity, and acceleration.
6. Physics parameters: High stiffness (k=0.1) and moderate dampening (d=0.8) to create bouncy, gelatinous text motion.
7. Mouse repulsion field: A virtual force field around the cursor (radius 150px) that applies a pushing force (magnitude 50) to particles, displacing them from their origins.
8. Auto-animating force: To ensure headless evaluation captures the effect, implement an invisible "virtual cursor" that moves in a Lissajous curve or figure-eight pattern across the text automatically.
9. Seamless looping background: Implement a pixel-based or shape-based 2D Simplex/Perlin noise field for the background, looping seamlessly every 10 seconds.
10. Color palette: A high-contrast cyberpunk palette (e.g., Deep Purple #0d0221, Neon Cyan #00f0ff, Magenta #ff003c, Bright Yellow #fcee0a).
11. Palette interpolation: The colors of the text particles and background must smoothly transition between 4 distinct color schemes over the 10-second loop.
12. Motion blur: Achieve trailing effects by clearing the background each frame with a low-opacity fill (e.g., `ctx.fillStyle = 'rgba(13, 2, 33, 0.2)'; ctx.fillRect()`) instead of `clearRect`.
13. Responsive resizing: Canvas must strictly listen to `resize` events and rebuild the text particle grid to remain perfectly centered at all times.
14. Connective geometry: Draw faint, low-opacity lines (0.1 alpha) between particles that are within 10 pixels of each other to create a web/plexus effect.
15. Time-based scaling: The overall scale of the text should pulsate gently (scale 1.0 to 1.05) using a sine wave based on `performance.now()`.

ARCHITECTURE / IMPLEMENTATION NOTES
Implement a clear Class structure in vanilla JavaScript. Define a `Particle` class with `update()` and `draw()` methods.
Use an Euler integration loop for the physics system. Ensure the update step is decoupled from the frame rate using a fixed timestep or delta-time multiplier to prevent physics explosions on high refresh rate monitors.
For the text sampling, create a secondary `<canvas>` not attached to the DOM, fill it with the target text using a system font like "Arial Black" or "Impact", read the raw `Uint8ClampedArray`, and map the bounding box to the center of the main display canvas.

POLISH
Easing functions: When transitioning between color palettes, use an ease-in-out cubic function rather than linear interpolation for a more cinematic feel.
Subtle grain/noise overlay: In the main `draw` loop, sprinkle random 1x1 pixel artifacts across the canvas (alpha 0.05) to give the poster a printed, tactile texture.
Text rendering detail: Instead of just drawing points, have the particles draw small directional line segments (velocity vectors) indicating their current motion path, enhancing the kinetic feel.

TECHNICAL CONSTRAINTS
Strictly a single HTML file. No external CSS, no GSAP, no Matter.js, no external font files.
Pure Canvas2D API.
Performance: Must maintain 60fps. Optimize the plexus line-drawing by utilizing spatial partitioning or limiting the distance checks to avoid O(N^2) slowdowns.
Screenshot Expectation at t=3s: The virtual cursor should be intersecting the typography, showing a dramatic, physics-based explosion/deformation of the "QWEN 3.8" letters. The text should appear as a dense web of glowing particles with motion blur trails clearly visible. The background must not be flat black, but exhibit the generative noise gradient.''')
    f.write('"""\n\n')

    # 04-shader-gallery
    f.write('GAMES["04-shader-gallery"] = """')
    f.write('''Domain: GLSL shader gallery.

MUST-HAVE FEATURES
1. WebGL 2.0 application hosting 4 entirely distinct, complex fragment shaders.
2. Shader 1: Fractal Brownian Motion (FBM) volumetric clouds with a sun scattering effect.
3. Shader 2: Raymarched Signed Distance Field (SDF) of morphing, twisting geometric primitives (e.g., a sphere morphing into a torus).
4. Shader 3: Cellular Voronoi organic tissue pattern with pulsing bioluminescent colors.
5. Shader 4: Synthwave neon grid landscape with a setting sun, mountains, and traveling grid lines.
6. UI Overlay: A beautiful HTML/CSS glassmorphism overlay containing 4 navigation buttons to switch between the shaders.
7. UI positioning: Anchored to the bottom center, styled with `backdrop-filter: blur(10px)`, semi-transparent background, and rounded borders.
8. Smooth transitions: When switching shaders, do not instantly swap. Implement a transition shader logic that crossfades (using `mix`) or wipe-transitions between the old and new shader over 1.5 seconds.
9. Uniform management: Pass `u_time`, `u_resolution`, `u_mouse`, and `u_transition_state` to all active shader programs smoothly.
10. Audio-reactive simulation: Since no actual audio is allowed, simulate audio reactivity. Create a JavaScript sine-wave oscillator based on time that injects a `u_audio_sync` uniform to make the shaders pulse (e.g., Voronoi cells expanding, synthwave sun vibrating).
11. CRT Overlay logic: A final post-processing pass (can be done in the same fragment shader or via a secondary framebuffer) that applies screen curvature, chromatic aberration, and scanlines.
12. Interactive mouse panning: `u_mouse` must actively displace the camera or coordinates in all 4 shaders (e.g., panning the FBM clouds).
13. Auto-rotation: To ensure the visual changes during headless evaluation, have the UI automatically switch to Shader 2 after 5 seconds if no interaction occurs.
14. UI Hover states: CSS `:hover` effects that expand the button, increase opacity, and add a glowing `box-shadow`.
15. Loading state: A brief 500ms black screen fading into the first shader to simulate a heavy system boot.

ARCHITECTURE / IMPLEMENTATION NOTES
Do not compile 4 massive monolithic shaders into one string if it causes branching performance issues. Instead, compile 4 separate WebGLProgram instances or a master program with `#ifdef` directives if you prefer passing a state uniform.
A state machine in JavaScript must handle the `currentShader`, `nextShader`, and `transitionProgress` variables.
During a transition, render both the current and next shader into offscreen Framebuffer Objects (FBOs), then render a final screen-spanning quad that blends the two textures based on `transitionProgress`.
The HTML UI should be z-indexed above the canvas.

POLISH
Click ripple effects: When a UI button is clicked, spawn a purely CSS-based expanding circle ripple inside the button.
CRT vignette: Darken the corners of the post-processing pass significantly to emphasize the retro monitor aesthetic.
Smooth zooming: Add a subtle, continuous 5% scale zoom-in to the canvas via CSS transform `scale(1.05)` over 10 seconds, resetting on shader switch.

TECHNICAL CONSTRAINTS
Single HTML file. Inline CSS (`<style>`), inline JS (`<script>`).
Pure WebGL. No Shadertoy embeds, no external images. All 4 shaders must be purely math-based (procedural).
Screenshot Expectation at t=3s: Shader 1 (FBM clouds) must be fully rendered with rich volumetric lighting. The post-processing CRT curve and scanlines must be clearly visible over the graphics. The glassmorphism UI panel must be centered at the bottom, perfectly styled, with "Shader 1" highlighted as active.''')
    f.write('"""\n\n')

