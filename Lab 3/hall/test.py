class TrafficLight:
    def __init__(self):
        self.colors = ['red', 'yellow', 'green']
        self.current_color = 'red'

    def change_color(self):
        current_index = self.colors.index(self.current_color)
        next_index = (current_index + 1) % len(self.colors)
        self.current_color = self.colors[next_index]

    def display_color(self):
        print(f"The current color is {self.current_color}")


# Usage example
traffic_light = TrafficLight()
traffic_light.display_color()  # Output: The current color is red
traffic_light.change_color()
traffic_light.display_color()  # Output: The current color is yellow
traffic_light.change_color()
traffic_light.display_color()  # Output: The current color is green
traffic_light.change_color()
traffic_light.display_color()  # Output: The current color is red