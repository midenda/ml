from math import log10, floor


class RangeError (Exception):

    def __init__ (self, errormessage):
        self.message = errormessage


def set_type (x, t = None, y = None):
    if type(t) is type:
        t = str(t)[8:-2]
    else:
        t = str (type(y))[8:-2]

    if t == "numpy.float64":
        t = "float"
    x = eval (f"{t}({x})")
    return x


def std_form (n):

    n = str (n)

    if n[0] == "-":
        sign = "-"
        n = n[1:]
    else:
        sign = ""

    i = n.find (".")
    e = n.find ("e")

    if i > 0:
        integer = n[:i].replace ("-", "")
        remainder = n[i+1:]
        for i in range (len (remainder)):
            if all (remainder[i:]) == "0":
                zeroes = "0"*len (remainder[i:])
                break
        else:
            zeroes = ""

    else:
        integer = n.replace ("-", "")
        remainder = ""
        zeroes = ""

    if integer == "":
        integer = "0"

    if e > 0:
        power = int (n[e+1:])
        digits = (integer + remainder)[:e-1]
        if remainder != "":
            constant = f"{integer[0]}.{digits[1:]}"
        else:
            constant = f"{integer[0]}"
    else:
        l = len (integer)
        digits = integer + remainder

        if l >= 1 and integer[0] != "0":
            power = len (digits[1:]) - len (remainder)
            constant = f"{integer[0]}.{digits[1:]}"

            if remainder == "":
                constant = constant.strip("0") + zeroes

        elif integer[0] == "0":
            for i in range (len (remainder)):
                if remainder[i] not in ("0", ".", ""):
                    constant = f"{remainder[i]}.{remainder[i+1:]}"
                    power = -1 - i
                    break
            else:
                constant = "0"
                power = 0

            if len (constant) == 2 and constant[1] == ".":
                constant = constant[0]

    sigfigs = len (constant.replace (".", ""))

    constant = sign + constant

    if power > 0:
        exponent = f" * 10**{power}"
        sf = f"{constant} * 10^{power}"
    elif power == 0:
        exponent = ""
        sf = constant
    else:
        exponent = f" * 10**({power})"
        sf = f"{constant} * 10^({power})"

    if float (constant) == int (float (constant)):
        constant = int (float (constant))
    else:
        constant = float (constant)

    return sf, sigfigs, constant, power, exponent


def get_dp (n) -> int:
    n = str (n)

    e = n.find ("e")

    if e > 0:
        power = int (n[e+1:])
        n = eval_exp (n[:e], power)

    i = n.find (".")

    if i > 0:
        return len (n[i+1:])
    else:
        return 0


def get_sf (n) -> int:
    return std_form (n)[1]


def get_power (n) -> int:
    return std_form (n)[3]


def eval_exp (constant, exponent = 0):

    sf = get_sf (constant)
    constant = str (constant)

    if constant[0] == "-":
        sign = "-"
        constant = constant[1:]
    else:
        sign = ""

    i = constant.find (".")
    e = constant.find ("e")

    if e > 0:
        power = int (constant[e+1:])
        constant = constant[:e]
    else:
        power = 0

    digits = constant.replace (".", "")

    if digits == "0":
        return "0"

    if i >= 0:


        if constant[0] == "0":
            for j in range (len (constant)):
                if constant[j] not in ("", ".", "0"):
                    currentpower = - (len (constant[i+1:j]) + 1) + power
                    break
            else:
                currentpower = 0
        else:
            currentpower = max (len (constant[:i]) - 1, 0) + power

        point = ""

        for j in range (len (constant)):
            if all(c == "0" for c in constant[j:]):
                num_0 = len (constant) - j
                break
        else:
            num_0 = 0

    else:
        i = len (digits) - 1
        currentpower = i + power
        point = "."
        num_0 = 0

    power = currentpower + exponent
    digits = digits.strip ("0")
    d = len (digits)


    if exponent < 0:
        exponent = 0

    if power > 0:
        i = power + 1

        if i >= d:
            j = i - d
            digits = digits + "0"*j
            if num_0 > j:
                digits = digits + "." + "0"*(num_0 - j)
            else:
                if sf > get_sf (digits) and num_0 - exponent > 0:
                    digits = digits + "." + "0"*(num_0 - exponent)

        elif i < d:
            digits = [c for c in digits]
            digits.insert(i, ".")
            digits = "".join (digits) + "0"*max (num_0 - exponent, 0)

    elif power == 0:
        digits = [c for c in digits]
        digits.insert(1, ".")
        digits = "".join (digits) + "0"*max (num_0 - exponent, 0)

    elif power < 0:

        i = power
        if i > 0:
            digits = [c for c in digits]
            digits.insert(i, ".")
            digits = "".join (digits) + "0"*max (num_0 - exponent, 0)
        elif i <= 0:
            digits = "0"*(-i) + digits + "0"*max (num_0 - exponent, 0)

            digits = [c for c in digits]
            digits.insert (1, ".")
            digits = "".join (digits)

    evaluated = digits.strip (".")

    return sign + evaluated


def round_up (n):
    l = [i for i in str(n)][::-1]

    sign = ""

    if l[0] == "-":
        l = l[1:]
        sign = "-"

    for i in range (len (l)):
        if l[i] != "9":
            l = ["0" for c in l[:i]] + [str (int (l[i]) + 1)] + l[i+1:]
            break

    else:
        l = ["0" for i in l] + ["1"]

    new_n = sign + "".join (l)[::-1]
    return new_n


def find_nearest (n, options):
    i = min ([(abs (float (n) - float (x)), index) for index, x in enumerate (options)])[1]
    return options[i]


def around (number, dp = None, sf = None, nearest = None, direction = None):

    t = type (number)

    sign = ""

    if str(number)[0] == "-":
        sign = "-"
        number = str(number)[1:]

    if sign == "-":
        if direction == "up":
            direction = "down"
        elif direction == "down":
            direction = "up"

    if dp != None:
        if dp >= 0:
            n = str (number)
            e = n.find ("e")

            exponent = std_form (n)[3]

            if e > 0:
                exponent = std_form (n)[3]
                n = n[:e]
                n = eval_exp (n, exponent)

            i = n.find(".")

            if i >= 0:
                l = len (n[i+1:])
                point = ""
            else:
                l = 0
                point = "."

            if dp > l:
                zeroes = "0"*(dp-l)
                rounded = f"{n}{point}{zeroes}"
            elif dp == l:

                rounded = n
            else:
                integer = n[:i]
                remainder = (n[i+1:])[:dp+1]

                r = len (remainder)

                if r > 1:
                    first_digit = int (remainder[0])
                    last_digit = int (remainder[-1])

                    if last_digit >= 5:
                        remainder = round_up (remainder[:-1])
                        if len (remainder) == r:
                            integer = round_up (integer)
                            remainder = "0"*dp

                    else:
                        remainder = remainder[:-1]

                    rounded = f"{integer}.{remainder}"

                else:
                    last_digit = int (remainder)

                    if last_digit >= 5:
                        rounded = round_up (integer)
                    else:
                        rounded = integer

        else:
            raise RangeError ("Argument is outside of valid range")

    elif sf != None:
        if sf > 0:
            if sf > 1:
                point = "."
            else:
                point = ""

            n = std_form (number)
            exponent = n[3]
            constant = str(n[2]).replace (".", "")
            if sf <= len(constant):
                constant = constant[:sf+1]
                dp = max (len (constant)-2, sf-1)
                constant = str (around (f"{constant[0]}.{constant[1:]}", dp = dp)).replace (".", "")

            else:
                constant = constant + "0"*(sf-len (constant))

            rounded = f"{constant[0]}{point}{constant[1:]}"

        else:
            raise RangeError ("Argument is outside of valid range")

        rounded = eval_exp (rounded, exponent)

    elif nearest != None:
        number = float (number)

        if type (nearest) in (int, float, str):
            nearest = [nearest]

        attempts = []

        for n in nearest:

            if n == int (n):
                n = int (n)

            remainder = (number % n)

            if direction is None:

                if abs (n - remainder) > n/2:
                    rounded = number - remainder
                else:
                    rounded = number + n - remainder

            elif direction == "up":
                rounded = number + n - remainder

            elif direction == "down":
                rounded = number - remainder

            rounded = around (rounded, dp = get_dp (n))

            if n == int (n):
                rounded = int (rounded)

            attempts.append (rounded)

        rounded = str (find_nearest (number, attempts))

    else:
        raise TypeError

    if float (rounded) == 0.0:
        return rounded

    return sign + rounded


def set_sf (n, sf):
    t = type (n)
    r = around (n, sf=sf)
    if t == float:
        try:
            if r == int (r):
                r = int (r)
        except ValueError:
            pass
    return r


if __name__ == '__main__':

    def run_tests ():
        from numpy import linspace

        l = [1234, 8999, 134139, 10090, 99]
        for c in l:
            result = round_up (c)
            print(f"Input:  {c}, \nOutput: {result}")

        l = [0.0006, 2.9999, .5, .51, 13718.35253, 0.42, 0.000080000001, 1.42,
             "340000", 3.00, "134.0000", 41000000000000, 41000000000000.0, 0.0000000000003, "0.00000000000030", "0.000000000000300",
             -0.0006, -2.9999, -.5, -.51, -13718.35253, -0.42, -0.000080000001, -1.42,
             "-340000", -3.00, "-134.0000", -41000000000000, -41000000000000.0, -0.0000000000003, "-0.00000000000030", "-0.000000000000300"]

        y = [x for x in linspace (0.1, 13, 42)]

        l = l + y + [-n for n in y]

        print ("\n\n --------------------------- ")

        for c in l:
            print(f"\n'around', 3dp:\nInput:  {c}, \nOutput: {around (c, dp = 3)} \n")
            print(f"\n'eval_exp', exp = -1:\nInput:  {c}, \nOutput: {eval_exp (c, -1)} \n")
            print(f"\n'set_sf', 1sf:\nInput:  {c}, \nOutput: {set_sf (c, 1)} \n")
            print(f"\n'std_form':\nInput:  {c}, \nOutput: {std_form (c)} \n")
            print(f"\n'around', nearest 10, up:\nInput:  {c}, \nOutput: {around (c, nearest = 10, direction = 'up')} \n")
            print(f"\n'around', nearest [4, 5], up:\nInput:  {c}, \nOutput: {around (c, nearest = [4, 5], direction = 'up')} \n")

    run_tests()
