from .bases import BinaryOperation


class Add(BinaryOperation):

    def forward(self, x):
        return x + self.value


class Sub(BinaryOperation):

    def forward(self, x):
        return x - self.value


class Mul(BinaryOperation):

    def forward(self, x):
        return x * self.value


class Div(BinaryOperation):

    def forward(self, x):
        return x / self.value
