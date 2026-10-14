import os
filepath = '~/B70-DOCS/scripts/tmp/b70-qwen38-games-harness-vision.py'

with open(filepath, 'a', encoding='utf-8') as f:
    # 09-tower-defense
    f.write('GAMES["09-tower-defense"] = """')
    f.write('''Domain: Tower-defense game.

MUST-HAVE FEATURES
1. Pure Canvas2D implementation of a grid-based map (e.g., 20x15 grid, 40px per tile).
2. Path generation: Define a specific, winding path of "dirt" tiles from a start node to an end node, surrounded by "grass" tiles.
3. Wave Manager: Spawns a stream of enemies (e.g., red circles) at the start point. Wave 1 spawns 10 enemies, Wave 2 spawns 15 faster enemies, etc.
4. Tower Type 1 (Laser): Fast firing (2 shots/sec), low damage, single target, infinite projectile speed (draws a line).
5. Tower Type 2 (Splash): Slow firing (0.5 shots/sec), high damage, spawns a projectile that travels and damages all enemies within a radius of the impact point.
6. Tower Type 3 (Frost): Low damage, but applies a status effect that halves enemy movement speed for 2 seconds.
7. Economy system: Start with 150 gold. Defeating enemies grants 10 gold. Towers cost 50, 100, and 75 gold respectively.
8. Placement mechanics: Clicking a valid grass tile while a tower type is selected in the UI places the tower.
9. Distance checking: Towers must scan for enemies within their radius using squared distance `(dx*dx + dy*dy <= r*r)` to avoid square root performance hits.
10. Turret tracking: The visual representation of the tower (e.g., a gun barrel) must rotate to face its current target smoothly using `Math.atan2`.
11. Projectile system: Update loop for projectiles traveling towards targets. If a target dies before impact, the projectile should travel to the last known location.
12. Particle explosions: When an enemy dies or a splash projectile hits, spawn a burst of 15 fading, scattering particles.
13. Health bars: Render small floating health bars above every enemy (green for healthy, red for damaged).
14. Base Health: The player has 20 lives. Enemies reaching the end of the path decrement this. Display "GAME OVER" when lives reach 0.
15. Auto-play boot: To ensure headless validation succeeds, the game must automatically start Wave 1, and procedurally auto-place one Laser tower and one Splash tower at optimal coordinates along the path at t=0.5s without user interaction.

ARCHITECTURE / IMPLEMENTATION NOTES
Structure the game using distinct lists for `enemies`, `towers`, `projectiles`, and `particles`.
Implement an overarching `Game` class with an `update(dt)` and `draw(ctx)` method.
Path following: Enemies should store an index of their current target waypoint on the path. When they reach it, increment the index and calculate the new velocity vector.

POLISH
Visual fidelity: Draw a subtle grid overlay. Make the towers distinct shapes (e.g., triangle for laser, hexagon for splash, diamond for frost).
UI integration: HTML overlay for the Gold, Lives, Wave number, and Tower selection buttons. Highlight the selected tower button.
Range indicator: When hovering over a valid placement tile, draw a faint translucent circle showing the tower's exact coverage radius.

TECHNICAL CONSTRAINTS
Single HTML file. Pure Canvas2D API. No external libraries or assets.
Screenshot Expectation at t=3s: The canvas must show the grid map and path. The auto-placed towers must be visible and actively firing at the first wave of enemies. Projectiles or lasers must be drawn on screen. Enemies must have visible health bars. UI showing gold and lives must be populated.''')
    f.write('"""\n\n')

    # 10-roguelike-dungeon
    f.write('GAMES["10-roguelike-dungeon"] = """')
    f.write('''Domain: Free choice (Turn-based roguelike).

MUST-HAVE FEATURES
1. Pure Canvas2D or DOM grid implementation. Top-down, strict turn-based movement on a tile grid (e.g., 40x30 tiles).
2. Procedural Dungeon Generation: Implement a BSP (Binary Space Partitioning) tree or Cellular Automata algorithm to generate a map of interconnected rectangular rooms and corridors, surrounded by solid wall tiles.
3. Field of View (FOV): Implement a shadowcasting or raycasting algorithm. The player only sees tiles within a radius of 8. Explored tiles outside this radius remain visible but are darkened (fog of war). Unexplored tiles are pure black.
4. Player Stats: HP (100), Attack (10), Defense (5), Level (1), XP (0). Displayed in an HTML UI panel.
5. Enemy AI: 3 distinct enemy types (e.g., Goblin/weak, Orc/medium, Troll/strong). Enemies only act after the player takes a turn. If they see the player (within FOV), they use A* pathfinding or simple step-towards logic to close the distance.
6. Melee Combat: Bumping into an enemy initiates an attack. Damage calculated as `max(1, Att - Def)`. The enemy then counter-attacks.
7. Message Log: A scrolling UI box at the bottom showing combat text (e.g., "You hit Orc for 5 dmg", "Orc dies and drops a potion").
8. Inventory system: Potions (heal 20 HP) and Swords (increase attack by 2) spawn procedurally on the floor. Bumping them picks them up. UI buttons exist to consume potions.
9. Smooth interpolation: Though logic is strictly grid-based, visually animate the player and enemies moving from tile A to tile B over 150ms using `lerp` for extreme polish.
10. Auto-explore script: For headless validation, the player character must autonomously take 10 turns (moving towards the nearest unexplored tile or attacking an adjacent enemy) immediately upon load, with a 250ms delay between turns.
11. ASCII / Retro styling: Draw the game using distinct colors, `fillRect`, and text characters (e.g., '@' for player, 'g' for goblin, '#' for walls) scaled up to emulate classic terminal roguelikes, but on a canvas.
12. Level transitions: A staircase spawns in the room furthest from the player. Stepping on it generates a new, harder floor (enemies have increased stats).
13. Health regeneration: Player regenerates 1 HP every 10 turns.
14. Death state: If player HP reaches 0, display a modal window with "YOU DIED" and the floor depth reached.
15. Pathfinding visualization: (Debug/Polish) Briefly highlight the tiles an enemy plans to walk on when they spot the player.

ARCHITECTURE / IMPLEMENTATION NOTES
Use an Entity-Component or deep OOP structure. `Map` class handles generation and FOV. `Actor` class handles stats, position, and drawing. `Scheduler` class handles the turn queue.
Wait for the movement animation (150ms) to finish before unlocking the player's keyboard input for the next turn to prevent visual desyncs.
Ensure the RNG (Random Number Generator) for dungeon creation is deterministic if a seed is provided.

POLISH
Screen shake: Add a subtle 50ms canvas translation shake when the player takes damage.
Blood splatters: When an actor dies, permanently paint the floor tile with a dark red background to indicate a battle occurred.
Log formatting: Color-code the message log (e.g., player damage taken in red, items found in green).

TECHNICAL CONSTRAINTS
Strictly a single HTML file. Pure Canvas2D API. No external libraries (no ROT.js).
Screenshot Expectation at t=3s: The auto-explore script will have executed a few turns. The screenshot must show the generated dungeon layout within the Field of View, the darkened explored areas, the player '@', at least one enemy or item in the room, and the message log populated with the initial automated actions.''')
    f.write('"""\n\n')

