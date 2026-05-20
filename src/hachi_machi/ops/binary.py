from .bases import BinaryOperation


class Add(BinaryOperation):
    """Addition operation."""

    def forward(self, x):
        return x + self.value


class Sub(BinaryOperation):
    """Subtraction operation."""

    def forward(self, x):
        return x - self.value


class Mul(BinaryOperation):
    """Multiplication operation."""

    def forward(self, x):
        return x * self.value


class Div(BinaryOperation):
    """Division operation."""

    def forward(self, x):
        return x / self.value
