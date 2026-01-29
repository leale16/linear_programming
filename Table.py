from abc import ABC
from typing import Union
import numpy as np
from fractions import Fraction
import copy


class Table(ABC):
    _history = []
    _class_type = "basic"

    def __init__(
        self,
        minmax: str,
        matrix: Union[list[list[str]], None] = None,
        basic_func: Union[list[str], None] = None,
    ) -> None:
        self.matrix, self.basic_func, self.minmax = matrix, basic_func, minmax
        self.table: np.ndarray = None
        self.width, self.length = 0, 0
        self._line = []
        self._column = []
        self.verios = [] # опорные элементы
        self.check_step = False # флаг завершения (нет допустимых шагов)

        self.set_full_task()

    def load_history(self):
        return self._history.pop()

    def record_history(self):
        self._history.append(self.copy())

    def copy(self):
        return copy.deepcopy(self)

    def get_basic_func(self): # коэфф. цел. ф-ии
        return self.basic_func

    def set_full_task(self):
        pass

    # выводим таблицу в читаемом формате
    def get_table(self):
        print("    ", end="")
        for i in self._line:
            print("{:>{}}".format(f"x{i}", 5), end=" ")
        print()
        for i in range(self.length):
            try:
                print(f"x{self._column[i]}  ", end="")
            except:
                print("f   ", end="")
            for j in range(self.width):
                print("{:>{}}".format(str(self.table[i, j]), 5), end=" ")
            print()

    def is_empty_history(self):
        return len(self._history) == 0

    # ищем все опорные элементы
    def serch(self):
        self.verios = []  # список допустимых опорных элементы
        for i in range(self.width - 1): # перебираем до b
            sum_num = -1
            indexes = [0, 0]
            if self.table[-1, i] < 0:  # F < 0
                for j in range(self.length - 1):
                    if self.table[j, i] > 0:
                        ratio = (
                            self.table[j, -1] / self.table[j, i]
                        )  # соотношение свободного члена к предполагаемому опорному элементу 
                        if sum_num == -1 or ratio < sum_num:
                            sum_num = ratio
                            indexes = [j, i]
                if sum_num != -1:
                    self.verios.append(indexes)
        self.check_step = len(self.verios) == 0 # хотя бы 1 оп.эл

    # сам симплекс-шаг
    def step(self, index_i, index_j):
        self._line[index_j], self._column[index_i] = (
            self._column[index_i],
            self._line[index_j],
        )
        # доп таблица под симплекс
        help_tab = np.zeros(shape=(self.length, self.width), dtype=Fraction)
        pivot = self.table[index_i, index_j] # из нынешней таблицы значение опорного элемента
        help_tab[index_i, index_j] = Fraction(1, pivot) # 1/выбр.опор.эл
        for j in range(self.width): # далее крест от 1/опорного элемента
            if j != index_j:
                help_tab[index_i, j] = self.table[index_i, j] / pivot
        for i in range(self.length):
            if i != index_i:
                help_tab[i, index_j] = -self.table[i, index_j] / pivot
        # (старые эл)-(эл над под оп.эл)*(строка с оп.эл из нынешней)
        for i in range(self.length):
            if i != index_i:
                for j in range(self.width):
                    if j != index_j:
                        help_tab[i, j] = (
                            self.table[i, j]
                            - self.table[i, index_j]
                            * self.table[index_i, j]
                            / pivot
                        )
        self.table = help_tab
        self.delete_column(index_j)

    # удаление колонки
    def delete_column(self, index):
        if index < 0 or index >= self.width:
            return
        # когда обе переменные небазисные удалить не можем
        if self._line[index] <= len(self.basic_func):
            return
        help_table = np.zeros(
            shape=(self.length, self.width - 1), dtype=Fraction
        )
        for j in range(index):
            help_table[:, j] = self.table[:, j]

        for j in range(index + 1, self.width):
            help_table[:, j - 1] = self.table[:, j]
        self._line.pop(index)
        self.width -= 1
        self.table = help_table

    # отриц.коэфф. в F ???
    def has_next_step(self):
        return any(val < 0 for val in self.table[-1, :-1])
 
    # проверка, что нет случая, когда задача неограничена снизу
    def check_table(self):
        for i in range(self.width):
            flag = True
            for j in self.table[:, i]:
                if j > 0:
                    flag = False
                    break
            if flag:
                return True
        return False

    def get_class_type(self):
        return self._class_type


class BasicTable(Table):
    def __init__(self, minmax, matrix=None, basic_func=None):
        super().__init__(minmax, matrix, basic_func)
    # строим таблицу
    def set_full_task(self):
        if self.minmax == "max":
            self.basic_func = [Fraction(-x) for x in self.basic_func]
        self.width = len(self.matrix[0])
        self.length = len(self.matrix)
        self._line = [(i + 1) for i in range(self.width - 1)]
        self._column = [(i + self.width) for i in range(self.length)]
        self.length += 1
        self.table = np.zeros(shape=(self.length, self.width), dtype=Fraction)
        for i in range(self.length - 1):
            row = self.matrix[i]
            # если своб.член > 0 оставляем, иначе *(-1)
            self.table[i] = row if row[-1] >= 0 else [-x for x in row]
        for i in range(self.width):
            self.table[-1, i] = -sum(self.table[: self.length - 1, i]) # -(сумма по столбцу)

    # проверка, что все последние элементы = 0
    def check_table(self):
        for i in self.table[-1, :-1]:
            if i != 0:
                return True
        return False
 
    # пересчет целевой функции
    def convert_to_simplex(self):
        basic_func = self.get_basic_func()
        for i in range(self.width):
            total = 0
            for j in range(self.length - 1):
                # каждый эл. столбца * на коэф.базиса при переменной и суммируем
                total += basic_func[self._column[j] - 1] * self.table[j, i]
            self.table[-1, i] = total
        for i in range(self.width - 1): # кроме своб.чл мен. знак
            self.table[-1, i] *= -1
        for i in range(len(self._line)): # добавляем коэф. при иксах
            self.table[-1, i] += basic_func[self._line[i] - 1]
        self.table[-1, -1] *= -1 # своб.чл
        return SimplexTable(
            self.table.copy(),
            basic_func,
            self._line,
            self._column,
            self.width,
            self.length,
        )


class SimplexTable(Table):
    def __init__(self, table, basic_func, line, column, width, length):
        self.table = table
        self.basic_func = basic_func
        self._line = line
        self._column = column
        self.width = width
        self.length = length
        self._class_type = "simplex"


if __name__ == "__main__":
    table = BasicTable("min", file_path="task4.json")
    table.get_table()
    table.serch()
    while not table.check_step:
        print(table.verios)
        i, j = [int(i) for i in input().split()]
        table.step(i, j)
        table.get_table()
        table.serch()
    print("Переход к симплекс методу")
    table = table.convert_to_simplex()
    while table.has_next_step() and not table.check_table():
        table.serch()
        print(table.verios)
        i, j = [int(i) for i in input().split()]
        table.step(i, j)
        table.get_table()
    if not table.has_next_step():
        print("answer")
        table.get_table()
    if table.check_table():
        print("wrong")
        table.get_table()
