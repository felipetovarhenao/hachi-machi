from .bases import RandomOperation


class AddRand(RandomOperation):
    """Random addition operation."""

    def forward(self, x):
        return x + self.func(x)


class SubRand(RandomOperation):
    """Random subtraction operation."""

    def forward(self, x):
        return x - self.func(x)


class MulRand(RandomOperation):
    """Random multiplication operation."""

    def forward(self, x):
        return x * self.func(x)


class DivRand(RandomOperation):
    """Random division operation."""

    def forward(self, x):
        return x / self.func(x)
