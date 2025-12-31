# yaps

yaps: Yet Another Pygame-Snake.

<img width="400" height="100%" alt="image" src="https://github.com/user-attachments/assets/33536150-2cad-4957-9e31-7016e6587aa3" />

## Running

`$ python yaps.py`

Defaults to autopilot enabled. Press `a` to toggle autopilot.

### Dependencies

See the [requirements file](./requirements.txt) for the full list. In short:

- PyGame powers the interactive UI.
- PyTorch is used by both the DQN trainer (`ml.py`) and the runtime ML autopilot
  (`ml_autopilot.py`).

Install everything with:

`$ pip install -r requirements.txt`

### Training an AI model

Use the standalone trainer to learn a policy without launching the PyGame window:

`$ python ml.py --episodes 1000 --cols 12 --rows 12`

The script saves the best-performing weights to `models/snake_dqn.pt` and prints
rolling statistics. Explore `python ml.py --help` for all tuning options (batch
size, epsilon schedule, board dimensions, etc.).

#### Using the trained autopilot in-game

Launch the main game with your checkpoint to let the ML policy drive the snake:

`$ python yaps.py --ml-model models/snake_dqn.pt --ml-hidden-size 256`

The `--ml-hidden-size` value must match what you used for training (the default
trainer width is 256). If the model cannot be loaded, YAPS automatically falls
back to the classic BFS-based autopilot.

### Controls

- Arrows for moving the snake.
- Space for resetting to default position.
- Click on a tile to jump to it.

### Customizing the grid

Override the defaults when launching the game:

- `--tile-size` for the pixel size of each tile.
- `--tiles-horizontal` for the number of columns.
- `--tiles-vertical` for the number of rows.

Example:

`$ python yaps.py --tile-size 64 --tiles-horizontal 14 --tiles-vertical 12`
