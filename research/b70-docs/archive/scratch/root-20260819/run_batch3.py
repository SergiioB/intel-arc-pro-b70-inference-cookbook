import importlib.util
import os

file_path = '~/B70-DOCS/scripts/tmp/b70-qwen38-games-harness-vision.py'
spec = importlib.util.spec_from_file_location('h', file_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
games = m.GAMES

updates = {
    "08-cyberpunk-terminal": {
        "gameplay": "Minute 0 to 1 centers on the automated boot sequence. The user watches the terminal type commands, feeling immersed in the hacker narrative. Mid-session (Minute 1 to 3): The user takes control, typing ls and cat. They discover restricted files that demand root access, prompting them to type exploit. The minigame triggers: a sudden intense burst where they must type 30 hex characters in 15 seconds. Mastery (Minute 3+): The user beats the minigame, gaining root access. They read the final classified logs, uncovering the full narrative. They begin testing standard UNIX commands out of curiosity, seeing how well the simulation holds up.\nEdge cases and exact expected outcomes:\n1. Typos in the exploit minigame: Typing the wrong character does not just ignore the input; it must flash the screen red, decrement the timer by 1 second as a penalty, and force the user to type the correct character to proceed.\n2. Backspace spamming: Holding backspace must stop exactly at the command prompt (user@sys:~$ ) and not delete the prompt itself.\n3. Rapid command execution: Spamming Enter on an empty line should cleanly reprint the prompt on a new line without breaking the DOM.\n4. Escaping directories: cd ../../../.. must clamp to the virtual root / and not throw undefined errors when attempting to access parent nodes that do not exist.\nDifficulty/Interest curve: Slow narrative burn (0-60s) -> sudden intense spike in difficulty during the exploit minigame -> satisfying resolution.",
        "arch": "Construct a TerminalEnv class. It manages a DOMRenderer and a FileSystem object.\nRender vs Simulation split:\n- Simulation: The FileSystem uses a nested JavaScript object { name: '/', type: 'dir', children: { ... } }. A pathResolver function handles parsing absolute and relative paths.\n- Render: A print(text, speed) function returns a Promise. It uses setTimeout to append characters to the DOM one by one. The global keyboard listener pushes to a buffer, which is immediately rendered to the active input line.\nState Machine states: BOOTING, IDLE_PROMPT (waiting for user input), EXECUTING_COMMAND, MINIGAME_ACTIVE.\nData Structures: A history array stores past commands, accessed via an index pointer that increments/decrements on Up/Down arrow keys.\nError Surface: Prevent XSS or DOM breakage. Any user input passed to cat or echo must be properly sanitized or inserted via textContent rather than innerHTML to avoid injecting raw HTML tags into the terminal.",
        "polish": "- Juice: When the user successfully completes the exploit minigame, trigger a dramatic CSS keyframe animation that rapidly flashes the text color from green to white and back over 200ms, accompanied by the text 'ACCESS GRANTED' typing out instantly.\n- Sound Design: Integrate a WebAudio OscillatorNode playing a tiny 10ms 'tick' (Square wave, 2000Hz, 0.05 gain) for every character printed to the screen, perfectly simulating an old mechanical relay or HDD seeking.\n- Hover/Focus states: Ensure the hidden input field always re-grabs focus if the user clicks anywhere on the document body (document.body.addEventListener('click', () => input.focus())).\n- Accessibility: Provide an HTML button outside the terminal styled as a 'Hardware Switch' to disable the CRT scanline and flicker animations for users sensitive to flashing lights.\n- Empty State / EOF: When the terminal boots, before any text is shown, the blinking cursor must blink alone for exactly 1.5 seconds to build anticipation."
    },
    "09-tower-defense": {
        "gameplay": "Minute 0 to 1: The user observes the auto-placed Laser and Splash towers easily dispatching the first wave of 10 slow enemies. They grasp the mechanics passively. Mid-session (Minute 1 to 3): Wave 3 introduces faster enemies. The user realizes they must build Frost towers to slow them down so the Splash towers can hit effectively. They spend their 150 gold strategically at chokepoints. Mastery (Minute 3+): The user is managing economy vs upgrades. They must quickly place/sell towers during the brief 5-second lull between waves to optimize their maze coverage against the massive Wave 5 (50 enemies).\nEdge cases and exact expected outcomes:\n1. Target out of range mid-flight: If a projectile is fired, but the enemy moves out of the tower's radius, the projectile must still track and hit the enemy. Range is only checked for the initial targeting phase.\n2. Enemy death before impact: If an enemy dies to a laser while a slow splash rocket is tracking it, the rocket must travel to the exact x, y coordinate where the enemy died and explode there, potentially damaging other enemies in the radius.\n3. Placing on path: Clicking a dirt path tile must trigger an 'Invalid Placement' visual (e.g., flash the tile red) and not deduct gold.\n4. Zero-health leak: If an enemy reaches the final waypoint with 0.1 health (due to a partial splash hit), it must die exactly on the threshold rather than leaking and decrementing player lives.\nDifficulty/Interest curve: Gentle slope for Wave 1-2. Sharp spike at Wave 4 when fast enemies demand Frost towers. Interest relies on the spatial puzzle of tower placement.",
        "arch": "Enforce a strict Module Pattern with a GameController handling the loop.\nRender vs Simulation split:\n- Simulation: update(dt) iterates through Towers (cooldown timers, distance checks), Enemies (move along path waypoints using normalized vectors), and Projectiles (AABB collision with targets). All logic is decoupled from framerate using dt.\n- Render: draw(ctx) simply iterates over the arrays. Towers draw their base, then ctx.save(); ctx.translate(); ctx.rotate(); to draw the turret barrel facing the target.\nState Machine states: MENU, BUILD_PHASE, COMBAT_PHASE, GAME_OVER.\nData Structures: The path should be an array of {x, y} waypoint objects. Enemies store a targetWaypointIndex.\nError Surface: Wrap the canvas context retrieval in a check. If getContext('2d') fails, inject an error message. Also, implement a strict object cap (e.g., max 500 particles) to prevent memory leaks during massive splash explosions.",
        "polish": "- Juice: When a Splash projectile hits, use a Cubic-Ease-Out curve to rapidly expand a translucent white circle ctx.arc() that fades to 0 opacity over 300ms, clearly telegraphing the damage radius.\n- Sound Design: WebAudio API. Laser tower: short high-pitch zap (Sawtooth, 800Hz -> 200Hz, 50ms). Splash tower explosion: White noise burst through a 100Hz Lowpass filter, decaying over 400ms.\n- Hover/Focus states: The UI tower selection buttons must highlight with a thick yellow border when selected. Hovering over a valid placement tile draws a green ghost of the tower; hovering over an invalid tile draws a red ghost.\n- Accessibility: Provide a 'Pause' button (or Spacebar toggle) that halts the COMBAT_PHASE update loop but still allows the player to place towers during the pause.\n- Empty State / Game Over: When lives hit 0, dim the entire canvas with a semi-transparent black overlay and slide in a 'DEFEAT' text from the top of the screen using a bounce easing effect."
    },
    "10-roguelike-dungeon": {
        "gameplay": "Minute 0 to 1 focuses on the auto-explore script, demonstrating the strict turn-based nature and the FOV lighting algorithm. The player watches the character map out the first few rooms autonomously. Mid-session (Minute 1 to 3): The user takes control via keyboard. They explore, finding a sword and a potion. They encounter a Troll (strong enemy) and realize they cannot win a straight fight. They tactically retreat into a narrow corridor to bottleneck the enemies, using the potion mid-fight to survive. Mastery (Minute 3+): The user understands the turn economy perfectly (1 step = 1 turn). They kite slower enemies, regenerate health safely in cleared rooms, and optimize their pathing to find the staircase down to Floor 2 with minimal combat.\nEdge cases and exact expected outcomes:\n1. Simultaneous movement block: If the player attempts to walk into a wall, the turn is NOT consumed. The system must wait for a valid action before letting enemies move.\n2. Diagonal corner cutting: If the grid prevents diagonal movement through walls, attempting to move diagonally across a solid corner must be blocked, or the game must strictly enforce 4-way movement.\n3. Item stacking: If an enemy dies on a tile that already contains a potion, the new dropped item should either overwrite it, bounce to an adjacent empty tile, or the tile must support an array of items.\n4. FOV asymmetry: The shadowcasting math must ensure that if the player can see an enemy, the enemy can see the player, preventing situations where the player is shot/spotted from the darkness.\nDifficulty curve: Floor 1 is trivial (Goblins). Floor 2 introduces Orcs. Floor 3 requires potions to survive. High replayability due to procedural generation.",
        "arch": "Implement using an Entity-Component or deep OOP structure, bounded within a single DOMContentLoaded wrapper.\nRender vs Simulation split:\n- Simulation: A central Scheduler acts as a state machine. The player inputs a move -> updates player grid x, y -> Scheduler loops through all Enemy entities -> enemies pathfind and update their grid x, y -> turn ends.\n- Render: A requestAnimationFrame loop runs independently of turns. It interpolates pixel coordinates from old_grid_pos to new_grid_pos over 150ms. Once the animation finishes, it unlocks the Scheduler to accept the next input.\nData Structures: Map is a 2D array of integers (0 = wall, 1 = floor). Entities are stored in a flat array but also mapped to a 2D spatial grid for O(1) collision lookups (e.g., entityGrid[y][x]).\nState Machine states: PLAYER_TURN, ANIMATING, ENEMY_TURN, GAME_OVER.\nError Surface: The BSP dungeon generation MUST contain a fail-safe iteration limit (e.g., max_splits = 10) to prevent infinite while loops from freezing the browser if a room cannot be placed.",
        "polish": "- Juice: When the player attacks, the sprite should quickly bump 50% into the target tile and snap back over 100ms. When taking damage, flash the player's sprite red for 1 frame.\n- Sound Design: Use WebAudio. Player step: very low, muted 'thud' (Triangle, 50Hz, 20ms). Combat hit: sharp metallic noise (Square wave, rapid frequency modulation). Level up: ascending major triad arpeggio.\n- Hover/Focus states: The UI inventory buttons (e.g., 'Use Potion') must clearly indicate they are clickable with a :hover background color change, and should grey out (opacity: 0.5, cursor: not-allowed) if the player has 0 potions.\n- Accessibility: Provide a 'High Visibility' toggle that changes the subtle darkened 'explored' tiles to bright gray and unexplored tiles to pure black, maximizing the contrast for the FOV edge.\n- Empty State: The message log should start with a welcoming, atmospheric string: 'You awaken in a cold, damp dungeon...', slowly fading in over 2 seconds."
    }
}

for key, mod in updates.items():
    if key not in games: continue
    text = games[key]
    
    parts = text.split("ARCHITECTURE / IMPLEMENTATION NOTES")
    part1 = parts[0]
    rest1 = parts[1]
    
    parts2 = rest1.split("POLISH")
    arch_text = parts2[0]
    rest2 = parts2[1]
    
    parts3 = rest2.split("TECHNICAL CONSTRAINTS")
    polish_text = parts3[0]
    tech_text = parts3[1]
    
    new_text = (
        part1.rstrip() + "\n\n" +
        "GAMEPLAY / EXPERIENCE DEPTH\n" +
        mod["gameplay"] + "\n\n" +
        "ARCHITECTURE / IMPLEMENTATION NOTES" +
        arch_text.rstrip() + "\n\n" +
        mod["arch"] + "\n\n" +
        "POLISH" +
        polish_text.rstrip() + "\n\n" +
        mod["polish"] + "\n\n" +
        "TECHNICAL CONSTRAINTS" +
        tech_text
    )
    games[key] = new_text

with open(file_path, 'w', encoding='utf-8') as f:
    f.write('# -*- coding: utf-8 -*-\n')
    f.write('"""\nQwen3.8-27B GPTQ vision-capable benchmark campaign harness.\nAgent setup: pi agent, off/medium thinking, 2026-08-16.\n10 ultra-detailed creative-coding prompts across domains (vision-verified campaign).\n"""\n\n')
    f.write('GAMES = {}\n\n')
    for k, v in games.items():
        f.write(f'GAMES["{k}"] = """{v}"""\n\n')

print("Batch 3 updated.")
