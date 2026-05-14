from multiads.assembly import ComponentOptions
from multiads.scenario.polars import POLAR_DEFAULT_SIZE


class SectionOptions(ComponentOptions):
    def __init__(
        self,
        polar: bool = False,
        polar_length: int = POLAR_DEFAULT_SIZE,
    ) -> None:
        self.polar = polar
        self.polar_length = polar_length
