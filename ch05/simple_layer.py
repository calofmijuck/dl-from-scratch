class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout


if __name__ == "__main__":
    apple, apple_num = 100, 2
    orange, orange_num = 150, 3
    tax = 1.1

    apple_layer = MulLayer()
    orange_layer = MulLayer()
    add_layer = AddLayer()
    tax_layer = MulLayer()

    apple_price = apple_layer.forward(apple, apple_num)
    orange_price = orange_layer.forward(orange, orange_num)
    total_price = add_layer.forward(apple_price, orange_price)
    price = tax_layer.forward(total_price, tax)

    dprice = 1
    dtotal_price, dtax = tax_layer.backward(dprice)
    dapple_price, dorange_price = add_layer.backward(dtotal_price)
    dorange, dorange_num = orange_layer.backward(dorange_price)
    dapple, dapple_num = apple_layer.backward(dapple_price)

    print(price)
    print(dapple_num, dapple, dorange, dorange_num, dtax)
