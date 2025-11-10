from encoded_motor_driver import EncodedMotorDriver
from machine import Timer


class WheelController(EncodedMotorDriver):
    def __init__(self, driver_ids, encoder_ids) -> None:
        super().__init__(driver_ids, encoder_ids)
        # Constants
        self.k_p = 1.2
        self.k_i = 0.0
        self.k_d = 1.5
        self.reg_vel_freq = 50  # Hz
        # Variables
        self.reg_vel_counter = 0
        self.duty = 0.0
        self.error = 0.0
        self.prev_error = 0.0
        self.error_inte = 0.0  # integral
        self.error_diff = 0.0  # differentiation
        self.ref_lin_vel = 0.0
        # PID controller config
        self.vel_reg_timer = Timer(
            freq=self.reg_vel_freq,
            mode=Timer.PERIODIC,
            callback=self.regulate_velocity,
        )

    def regulate_velocity(self, timer):
        if self.ref_lin_vel == 0.0 or self.reg_vel_counter > self.reg_vel_freq:
            self.stop()
            self.prev_error = 0.0
        else:
            self.error = self.ref_lin_vel - self.meas_lin_vel  # ang_vel also works
            self.error_inte += self.error
            self.error_diff = self.error - self.prev_error
            self.prev_error = self.error  # UPDATE previous error
            inc_duty = (
                self.k_p * self.error
                + self.k_i * self.error_inte
                + self.k_d * self.error_diff
            )
            self.duty = self.duty + inc_duty
            if self.duty > 0:
                if self.duty > 1.0:
                    self.duty = 1.0
                self.forward(self.duty)
            else:
                if self.duty < -1.0:
                    self.duty = -1.0
                self.backward(-self.duty)
            self.reg_vel_counter += 1

    def set_wheel_velocity(self, ref_lin_vel):
        if ref_lin_vel is not self.ref_lin_vel:
            self.ref_lin_vel = ref_lin_vel
            self.prev_error = 0.0
            self.error_inte = 0.0
            self.reg_vel_counter = 0


if __name__ == "__main__":
    """ Use following tuning PID"""
    from utime import sleep
    from machine import Pin

    wc = WheelController(
        driver_ids=(15, 13, 14),
        encoder_ids=(26, 27),
    )
    STBY = Pin(12, Pin.OUT)
    STBY.on()
    while True:
        
        wc.set_wheel_velocity(.65)
        print(
            f"Reference velocity={wc.ref_lin_vel} m/s, Measured velocity={wc.meas_lin_vel} m/s"
        )
        sleep(0.01)
    # Terminate
    STBY.off()
