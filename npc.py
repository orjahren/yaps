from player import Player

from helpers import DIRECTIONS, direction_change_is_legal, get_key_from_value


class Npc(Player):
    def __init__(self, surface, width, height):
        super().__init__(surface, width, height)

    def _get_next_direction(self, fruit_coords):

        if fruit_coords:
            fruit_x, fruit_y = fruit_coords
            npc_x, npc_y = self.get_pos()

            if fruit_x > npc_x:
                return DIRECTIONS["RIGHT"]
            elif fruit_x < npc_x:
                return DIRECTIONS["LEFT"]
            elif fruit_y > npc_y:
                return DIRECTIONS["DOWN"]
            elif fruit_y < npc_y:
                return DIRECTIONS["UP"]

   # NOTE: Overriding Player's update method to implement NPC movement logic
    def update(self, delta_ms, fruit_coords=None):
        """Advance the player when enough time has elapsed."""
        self._move_accumulator += delta_ms
        if self._move_accumulator < self._move_delay:
            return
        # Preserve leftover time so slight jitter is averaged out
        self._move_accumulator %= self._move_delay

        print("NPC is moving")
        # print frit_coords for debugging
        if fruit_coords:
            print(f"Fruit at: {fruit_coords}")

        next_direction = self._get_next_direction(fruit_coords)
        if direction_change_is_legal(self.get_direction(), next_direction):
            print(
                f"NPC changing direction to {get_key_from_value(DIRECTIONS, next_direction)}")
            self.set_direction(next_direction)

        # Simple NPC logic: change direction randomly at intersections
        self.move()
