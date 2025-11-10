from motor_driver import MotorDriver
from machine import Pin, Timer
from math import pi


class EncodedMotorDriver(MotorDriver):
    def __init__(self, driver_ids, encoder_ids) -> None:
        super().__init__(*driver_ids)
        # Pin configuration
        self.enc_a_pin = Pin(encoder_ids[0], Pin.IN)
        self.enc_b_pin = Pin(encoder_ids[1], Pin.IN)
        self.enc_a_pin.irq(
            trigger=Pin.IRQ_RISING | Pin.IRQ_FALLING, handler=self._update_counts_a
        )
        self.enc_b_pin.irq(
            trigger=Pin.IRQ_RISING | Pin.IRQ_FALLING, handler=self._update_counts_b
        )
        # Variables
        self._enc_a_val = self.enc_a_pin.value()
        self._enc_b_val = self.enc_b_pin.value()
        self.encoder_counts = 0
        self.prev_counts = 0
        self.meas_ang_vel = 0.0
        self.meas_lin_vel = 0.0
        # Constants
        self.meas_radius = 0.025  # m
        self.gear_ratio = 98.5
        self.cpr = 28  # CPR = PPR * 4
        self.vel_meas_freq = 100  # Hz
        # Timer configuration
        self.vel_meas_timer = Timer(
            freq=self.vel_meas_freq,
            mode=Timer.PERIODIC,
            callback=self.measure_velocity,
        )

    def _update_counts_a(self, pin):
        self._enc_a_val = pin.value()
        if self._enc_a_val == 1:
            if self._enc_b_val == 0:  # a=1, b=0
                self.encoder_counts += 1
            else:  # a=1, b=1
                self.encoder_counts -= 1
        else:
            if self._enc_b_val == 0:  # a=0, b=0
                self.encoder_counts -= 1
            else:  # a=0, b=1
                self.encoder_counts += 1

    def _update_counts_b(self, pin):
        self._enc_b_val = pin.value()
        if self._enc_b_val == 1:
            if self._enc_a_val == 0:  # b=1, a=0
                self.encoder_counts -= 1
            else:  # b=1, a=1
                self.encoder_counts += 1
        else:
            if self._enc_a_val == 0:  # b=0, a=0
                self.encoder_counts += 1
            else:  # b=0, a=1
                self.encoder_counts -= 1

    def reset_encoder_counts(self):
        self.encoder_counts = 0

    def measure_velocity(self, timer):
        delta_counts = self.encoder_counts - self.prev_counts
        self.prev_counts = self.encoder_counts  # UPDATE prev_counts
        counts_per_sec = delta_counts * self.vel_meas_freq  # delta_c / delta_t
        orig_rev_per_sec = counts_per_sec / self.cpr
        orig_rad_per_sec = orig_rev_per_sec * 2 * pi  # original motor shaft velocity
        self.meas_ang_vel = orig_rad_per_sec / self.gear_ratio
        self.meas_lin_vel = self.meas_ang_vel * self.meas_radius


# TEST
if __name__ == "__main__":  # Test only the encoder part
    from time import sleep

    # SETUP
    # emd = EncodedMotorDriver(
    #     driver_ids=(9, 11, 10),
    #     encoder_ids=(16, 17),
    # )  # channel A motor, encoder's green and yellow on GP16 and GP17
    emd = EncodedMotorDriver(
        driver_ids=(15, 13, 14),
        encoder_ids=(18, 19),
    )  # channel B motor, encoder's green and yellow on GP18 and GP19
    STBY = Pin(12, Pin.OUT)
    STBY.off()

    # LOOP
    STBY.on()  # enable motor driver
    # Forwardly ramp up and down
    for i in range(100):
        emd.forward((i + 1) / 100)
        #         print(f"f, dc: {i}%, enc_cnt: {emd.encoder_counts}")
        print(
            f"meas's angular velocity={emd.meas_ang_vel}, linear velocity={emd.meas_lin_vel}"
        )
        sleep(4 / 100)  # 4 seconds to ramp up
    for i in reversed(range(100)):
        emd.forward((i + 1) / 100)
        #         print(f"f, dc: {i}%, enc_cnt: {emd.encoder_counts}")
        print(
            f"meas's angular velocity={emd.meas_ang_vel}, linear velocity={emd.meas_lin_vel}"
        )
        sleep(4 / 100)  # 4 seconds to ramp down
    # Backwardly ramp up and down
    for i in range(100):
        emd.backward((i + 1) / 100)
        #         print(f"b, dc: {i}%, enc_cnt: {emd.encoder_counts}")
        print(
            f"meas's angular velocity={emd.meas_ang_vel}, linear velocity={emd.meas_lin_vel}"
        )
        sleep(4 / 100)  # 4 seconds to ramp up
    for i in reversed(range(100)):
        emd.backward((i + 1) / 100)
        #         print(f"b, dc: {i}%, enc_cnt: {emd.encoder_counts}")
        print(
            f"meas's angular velocity={emd.meas_ang_vel}, linear velocity={emd.meas_lin_vel}"
        )
        sleep(4 / 100)  # 4 seconds to ramp down

    # Terminate
    emd.stop()
