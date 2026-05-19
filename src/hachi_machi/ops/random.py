from .bases import RandomOperation


class AddRand(RandomOperation):

    def forward(self, x):
        return x + self.func(x)


class SubRand(RandomOperation):

    def forward(self, x):
        return x - self.func(x)


class MulRand(RandomOperation):

    def forward(self, x):
        return x * self.func(x)


class DivRand(RandomOperation):

    def forward(self, x):
        return x / self.func(x)
