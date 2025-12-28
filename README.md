# yaps

yaps: Yet Another Pygame-Snake.

<img width="400" height="100%" alt="image" src="https://github.com/user-attachments/assets/33536150-2cad-4957-9e31-7016e6587aa3" />

## Running

`$ python yaps.py`

Defaults to autopilot enabled. Press `a` to toggle autopilot.

### Dependencies

PyGame. See the [requirements file](./requirements.txt).

`$ pip install -r requirements.txt`

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
