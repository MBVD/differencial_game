class Driver:
    def __init__(self):
        self.x = 300.0
        self.y = 500.0
        self.theta = 0.0
        self.phi = 0.0

        self.phi_step = 0.05
        self.phi_return = 0.03

    def update(self, keys):
        left_pressed = ("a" in keys) or ("A" in keys)
        right_pressed = ("d" in keys) or ("D" in keys)

        if left_pressed and not right_pressed:
            self.phi -= self.phi_step
        elif right_pressed and not left_pressed:
            self.phi += self.phi_step
        else:
            if self.phi > 0:
                self.phi -= self.phi_return
                if self.phi < 0:
                    self.phi = 0.0
            elif self.phi < 0:
                self.phi += self.phi_return
                if self.phi > 0:
                    self.phi = 0.0

        if self.phi < -1.0:
            self.phi = -1.0
        if self.phi > 1.0:
            self.phi = 1.0