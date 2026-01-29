import sys

import numpy as np
import json
from fractions import Fraction
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QComboBox,
    QMessageBox,
    QSpinBox,
    QTabWidget,
    QFileDialog,
    QScrollArea,
    QDialog 
)
from PyQt6.QtGui import QPixmap, QFont, QPalette, QColor
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from shapely.geometry import LineString, Polygon
import matplotlib.pyplot as plt
from Table import BasicTable
from random import randint

# виджет для отображения графика
class GraphicalMethodCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

# для отображения процесса решения симплекса
class SimplexWindow(QDialog):
    def __init__(
        self, parent, basic_func, constraints, minimize, use_fractions
    ):
        super().__init__(parent)
        self.setWindowTitle("Симплекс-метод")
        self.resize(1000, 600)

        self.there_is_no_wrong = True
        self.basic_func = basic_func
        self.constraints = constraints
        self.minimize = minimize
        self.use_fractions = use_fractions

        self.phase = "basic"
        self.auto_step_index = None

        self.layout = QVBoxLayout(self)

        self.info_label = QLabel("Базисная таблица")
        self.layout.addWidget(self.info_label)

        self.table_widget = QTableWidget()
        self.layout.addWidget(self.table_widget)

        btn_layout = QHBoxLayout()
        self.next_btn = QPushButton("Следующий шаг")
        self.next_btn.clicked.connect(self.auto_step)
        btn_layout.addWidget(self.next_btn)

        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)

        self.layout.addLayout(btn_layout)

        self._init_basic_table()

        self.back_btn = QPushButton("Назад")
        self.back_btn.clicked.connect(self.undo_step)
        self.back_btn.setEnabled(False)
        btn_layout.insertWidget(0, self.back_btn)
        self._update_view()

    # создаем базовую таблицу из ограничений
    def _init_basic_table(self):
        matrix = []
        # преобразуем список ограничений в матричный формат
        for c in self.constraints:
            matrix.append(c["coeff"] + [c["value"]])

        self.table_model = BasicTable(
            minmax="min" if self.minimize else "max",
            matrix=matrix,
            basic_func=self.basic_func,
        )
    
    # прорисовка таблицы
    def _update_view(self):
        table = self.table_model.table
        rows, cols = table.shape
        self.back_btn.setEnabled(not self.table_model.is_empty_history())

        self.table_widget.clear()
        self.table_widget.setRowCount(rows)
        self.table_widget.setColumnCount(cols)

        self.table_widget.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )

        # заголовки столбцов
        for j in range(cols - 1):
            self.table_widget.setHorizontalHeaderItem(
                j, QTableWidgetItem(f"x{self.table_model._line[j]}")
            )
        self.table_widget.setHorizontalHeaderItem(
            cols - 1, QTableWidgetItem("b")
        )

        # заголовки строк
        for i in range(rows - 1):
            self.table_widget.setVerticalHeaderItem(
                i, QTableWidgetItem(f"x{self.table_model._column[i]}")
            )
        self.table_widget.setVerticalHeaderItem(
            rows - 1, QTableWidgetItem("f")
        )

        for i in range(rows):
            for j in range(cols):
                item = QTableWidgetItem(
                    str(table[i, j])
                    if self.use_fractions
                    else str(round(float(table[i, j]), 2))
                )
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                self.table_widget.setItem(i, j, item)

        self._highlight_cells()

    # отрисовка цветами опорных элементов
    def _highlight_cells(self):
        self.table_model.serch()

        for i, j in self.table_model.verios:
            item = self.table_widget.item(i, j)
            item.setBackground(QColor(255, 215, 0))  # жёлтый
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable # активна/выделять
            )

        if self.table_model.verios:
            index = self.table_model.verios[
                randint(0, len(self.table_model.verios) - 1)
            ]
            item = self.table_widget.item(index[0], index[1])
            item.setBackground(QColor(0, 200, 0))  # зелёный
            self.auto_step_index = index

        self.table_widget.cellClicked.connect(self.manual_step)

    # обработчик клика по ячейке таблицы
    def manual_step(self, row, col):
        if self.table_model.verios:
            if [row, col] not in self.table_model.verios:
                return
            self.auto_step_index = None
            self._do_step(row, col)

    # обработчик кнопки "Следующий шаг" автоматически
    def auto_step(self):
        if not self.table_model.verios:
            return
        self._do_step(self.auto_step_index[0], self.auto_step_index[1])

    # основной метод выполнения симплекс-шага
    def _do_step(self, i, j):
        if self.there_is_no_wrong:
            self.table_model.record_history()
            self.auto_step_index = None

            self.table_model.step(i, j)

            if self.phase == "basic" and not self.table_model.has_next_step():
                if self.table_model.check_table():
                    self.there_is_no_wrong = False
                    QMessageBox.critical(
                        self, "Ошибка", "Задача не имеет решения"
                    )
                    return
                self.table_model = self.table_model.convert_to_simplex()
                self.phase = "simplex"
                self.info_label.setText("Симплекс-таблица")

            if self.phase == "simplex":
                if not self.table_model.has_next_step():
                    self._show_answer()
                    return
                if self.table_model.check_table():
                    self.there_is_no_wrong = False
                    QMessageBox.critical(
                        self, "Ошибка", "Задача не имеет решения"
                    )
                    return

        self._update_view()

    # метод для отмены последнего шага
    def undo_step(self):
        self.there_is_no_wrong = True
        self.table_model = self.table_model.load_history()
        self.phase = self.table_model.get_class_type()

        if self.table_model.is_empty_history():
            self.back_btn.setEnabled(False)

        self.info_label.setText(
            "Базисная таблица" if self.phase == "basic" else "Симплекс-таблица"
        )

        self._update_view()

    # метод для отображения оптимального решения
    def _show_answer(self):
        answer_vars = [0] * len(self.basic_func)
        for i in range(len(self.table_model._column)): # по базисным
            answer_vars[self.table_model._column[i] - 1] = (
                self.table_model.table[i, -1]
            )
        text = "Оптимальное решение:\nx* = ("
        for var in answer_vars[:-1]:
            text += f"{var}, "
        text += f"{answer_vars[-1]})\n"
        if self.minimize:
            text += (
                f"\nЗначение целевой функции: F={-self.table_model.table[-1, -1]}"
            )
        else:
            text += f"\nЗначение целевой функции: F={self.table_model.table[-1, -1]}"
        QMessageBox.information(self, "Решение", text)
        self._update_view()


class LinearProgrammingApp(QMainWindow):
    def set_dark_theme(self):
        dark_palette = QPalette()

        # Настройка палитры
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53)) # темно-серый
        dark_palette.setColor(
            QPalette.ColorRole.WindowText, Qt.GlobalColor.white
        )
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35)) # еще темнее
        dark_palette.setColor(
            QPalette.ColorRole.AlternateBase, QColor(53, 53, 53)
        )
        dark_palette.setColor(
            QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white
        )
        dark_palette.setColor(
            QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white
        )
        dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark_palette.setColor(
            QPalette.ColorRole.ButtonText, Qt.GlobalColor.white
        )
        dark_palette.setColor(
            QPalette.ColorRole.BrightText, Qt.GlobalColor.red
        )
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        dark_palette.setColor(
            QPalette.ColorRole.Highlight, QColor(42, 130, 218)
        )
        dark_palette.setColor(
            QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black
        )

        # Применяем палитру
        self.setPalette(dark_palette)

        # Стиль для QTableWidget
        self.setStyleSheet(
            """
             /* Стиль для таблиц QTableWidget (симплекс-таблицы) */
            QTableWidget {
                background-color: rgb(45, 45, 45);
                color: white;
                gridline-color: rgb(80, 80, 80);
            }
            /* Стиль для заголовков столбцов и строк таблицы */
            QHeaderView::section {
                background-color: rgb(53, 53, 53);
                color: white;
                padding: 4px;
                border: 1px solid rgb(80, 80, 80);
            }
            /* Стиль для панели вкладок */
            QTabWidget::pane {
                border: 1px solid rgb(80, 80, 80);
            }
            /* Стиль для отдельных вкладок */
            QTabBar::tab {
                background: rgb(53, 53, 53);
                color: white;
                padding: 8px;
                border: 1px solid rgb(80, 80, 80);
            }
            /* Стиль для выбранной (активной) вкладки */
            QTabBar::tab:selected {
                background: rgb(35, 35, 35);
            }
        """
        )

        # Стиль для matplotlib (графика)
        plt.style.use("dark_background")
        self.canvas.fig.set_facecolor("#2D2D2D")
        self.canvas.axes.set_facecolor("#2D2D2D")
        self.setStyleSheet(
            """
        QTabWidget {
        background: rgb(53, 53, 53);
        border: 1px solid rgb(80, 80, 80);
        }

        /* Панель с ярлыками вкладок */
        QTabWidget::pane {
            border-top: 2px solid rgb(60, 60, 60);
            position: absolute;
            top: -1px;
            background: rgb(45, 45, 45);
        }

        /* Отдельные вкладки */
        QTabBar::tab {
            background: rgb(70, 70, 70);
            color: rgb(220, 220, 220);
            padding: 8px 12px;
            margin-right: 2px;
            border: 1px solid rgb(90, 90, 90);
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }

        /* Активная вкладка */
        QTabBar::tab:selected {
            background: rgb(45, 45, 45);
            color: white;
            border-color: rgb(110, 110, 110);
            border-bottom: 1px solid rgb(45, 45, 45); 
        }

        /* Неактивная вкладка при наведении */
        QTabBar::tab:hover:!selected {
            background: rgb(80, 80, 80);
        }

        /* Правый угол вкладок (для RTL тоже) */
        QTabBar::tab:first:selected {
            margin-left: 0;
        }
        QTabBar::tab:last:selected {
            margin-right: 0;
        }
        QTabBar::tab:only-one {
            margin: 0;
        }
                /* Общие стили */
                QWidget {
                    background-color: rgb(53, 53, 53);
                    color: white;
                    selection-background-color: rgb(42, 130, 218);
                    selection-color: black;
                }

                /* Стили кнопок */
                QPushButton {
                    background-color: rgb(70, 70, 70);
                    border: 1px solid rgb(90, 90, 90);
                    border-radius: 4px;
                    padding: 5px;
                    min-width: 80px;
                }

                QPushButton:hover {
                    background-color: rgb(80, 80, 80);
                    border: 1px solid rgb(100, 100, 100);
                }

                QPushButton:pressed {
                    background-color: rgb(60, 60, 60);
                    border: 1px solid rgb(70, 70, 70);
                }

                QPushButton:disabled {
                    background-color: rgb(50, 50, 50);
                    color: rgb(150, 150, 150);
                }

                /* Стили таблиц */
                QTableWidget {
                    background-color: rgb(45, 45, 45);
                    color: white;
                    gridline-color: rgb(80, 80, 80);
                }
            """
        )

        # Стиль для matplotlib
        plt.style.use("dark_background")
        self.canvas.fig.set_facecolor("#2D2D2D")
        self.canvas.axes.set_facecolor("#2D2D2D")

    # конструктор класса
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Методы оптимизации")
        self.setGeometry(100, 100, 1000, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        self.init_ui()
        self.set_dark_theme()

    # инициализация пользовательского интерфейса
    def init_ui(self):
        # создаем вкладки
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.input_tab = QWidget()
        self.tabs.addTab(self.input_tab, "Ввод данных")
        self.input_layout = QVBoxLayout()
        self.input_tab.setLayout(self.input_layout)

        self.problem_type_layout = QHBoxLayout()
        self.problem_type_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.problem_type_label = QLabel("Тип задачи:")
        self.problem_type_combo = QComboBox()
        self.problem_type_combo.addItems(["Максимизация", "Минимизация"])

        self.format_label = QLabel("Формат чисел:")
        self.format_combo = QComboBox()
        self.format_combo.addItems(["Десятичные", "Обыкновенные дроби"])

        self.problem_type_layout.addWidget(self.problem_type_label)
        self.problem_type_layout.addWidget(self.problem_type_combo)
        self.problem_type_layout.addSpacing(20)
        self.problem_type_layout.addWidget(self.format_label)
        self.problem_type_layout.addWidget(self.format_combo)
        self.problem_type_layout.addStretch()  # выравнивание по левому краю

        self.input_layout.addLayout(self.problem_type_layout)

        # настройка количества переменных и ограничений
        self.vars_constraints_container = QWidget()
        self.vars_constraints_layout = QVBoxLayout(
            self.vars_constraints_container
        )

        self.vars_layout = QVBoxLayout()
        self.vars_label = QLabel("Количество переменных:")
        self.vars_spin = QSpinBox()
        self.vars_spin.setMinimum(1)
        self.vars_spin.setMaximum(16)
        self.vars_spin.setFixedSize(80, 25)
        self.vars_spin.setValue(2)
        self.vars_layout.addWidget(self.vars_label)
        self.vars_layout.addWidget(self.vars_spin)

        self.constraints_layout = QVBoxLayout()
        self.constraints_label = QLabel("Количество ограничений:")
        self.constraints_spin = QSpinBox()
        self.constraints_spin.setMinimum(1)
        self.constraints_spin.setValue(3)
        self.constraints_spin.setFixedSize(80, 25)
        self.constraints_layout.addWidget(self.constraints_label)
        self.constraints_layout.addWidget(self.constraints_spin)

        self.vars_constraints_layout.addLayout(self.vars_layout)
        self.vars_constraints_layout.addLayout(self.constraints_layout)
        self.vars_constraints_container.setFixedWidth(200)
        self.input_layout.addWidget(self.vars_constraints_container)

        # кнопка для создания таблиц
        self.create_tables_btn = QPushButton("Создать таблицы")
        self.create_tables_btn.clicked.connect(self.create_tables)
        self.input_layout.addWidget(self.create_tables_btn)

        # таблица для целевой функции 
        self.objective_layout = QHBoxLayout()
        self.objective_label = QLabel("Целевая функция (коэффициенты):")
        self.objective_table = QTableWidget(1, 1)
        self.objective_table.horizontalHeader().setVisible(False)
        self.objective_table.verticalHeader().setVisible(False)
        self.objective_table.setMaximumHeight(50) 

        self.objective_layout.addWidget(self.objective_label)
        self.objective_layout.addWidget(self.objective_table)
        self.input_layout.addLayout(self.objective_layout)

        # таблица для ограничений
        self.constraints_label = QLabel("Ограничения:")
        self.constraints_table = QTableWidget()
        self.constraints_table.horizontalHeader().setVisible(False)
        self.constraints_table.verticalHeader().setVisible(False)
        self.input_layout.addWidget(self.constraints_label)
        self.input_layout.addWidget(self.constraints_table)

        self.solve_buttons_layout = QHBoxLayout()
        self.solve_btn = QPushButton("Решить графически")
        self.simplex_btn = QPushButton("Решить симплекс-методом")

        self.solve_btn.clicked.connect(self.solve_problem)
        self.simplex_btn.clicked.connect(self.open_simplex_window)

        self.solve_buttons_layout.addWidget(self.solve_btn)
        self.solve_buttons_layout.addWidget(self.simplex_btn)
        self.input_layout.addLayout(self.solve_buttons_layout)

        self.load_btn = QPushButton("Загрузить из файла")
        self.load_btn.clicked.connect(self.load_from_file)
        self.main_layout.addWidget(self.load_btn)

        self.result_tab = QWidget()
        self.tabs.addTab(self.result_tab, "Результаты")
        self.result_layout = QVBoxLayout()
        self.result_tab.setLayout(self.result_layout)

        # график
        self.canvas = GraphicalMethodCanvas(self, width=8, height=6)
        self.result_layout.addWidget(self.canvas)

        # текстовые результаты
        self.result_output = QLabel("")
        self.result_output.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.result_output.setStyleSheet(
            "border: 1px solid gray; padding: 5px;"
        )
        self.result_layout.addWidget(self.result_output)

        self.create_tables_btn.setProperty("class", "important")
        self.solve_btn.setProperty("class", "important")
        self.simplex_btn.setProperty("class", "important")
        self.update_button_styles()

        # кнопка сохранения в JSON
        self.save_json_btn = QPushButton("Сохранить в JSON")
        self.save_json_btn.clicked.connect(self.save_to_json)
        self.main_layout.addWidget(self.save_json_btn)

        # создаем вкладку со справкой
        self.reference_tab = QWidget()
        self.tabs.addTab(self.reference_tab, "Справка")

        main_layout = QVBoxLayout(self.reference_tab)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ScrollArea для прокрутки
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #2D2D2D;
                border: none;
            }
        """)

        # создаем контейнер для содержимого
        content_widget = QWidget()
        content_widget.setStyleSheet("background-color: #2D2D2D;")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(20)
        content_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("📋 Руководство пользователя")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #42a2da;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(title_label)

        content_layout.addSpacing(20)

        # функция для загрузки изображений с абсолютным путем
        def load_image(image_path):
            try:
                import os
                # проверяем разные возможные пути
                paths_to_try = [
                    image_path,
                    os.path.join("photo", image_path),
                    os.path.join(os.path.dirname(__file__), "photo", image_path),
                    os.path.join(os.getcwd(), "photo", image_path)
                ]
                
                for path in paths_to_try:
                    if os.path.exists(path):
                        pixmap = QPixmap(path)
                        if not pixmap.isNull():
                            # print(f"Изображение загружено: {path}")
                            return pixmap
                        else:
                            print(f"Не удалось загрузить изображение: {path}")
                
                print(f"Изображение не найдено: {image_path}")
                return QPixmap()
            except Exception as e:
                print(f"Ошибка при загрузке изображения {image_path}: {e}")
                return QPixmap()

        step1_label = QLabel("1. Ввод параметров задачи")
        step1_label.setStyleSheet("color: #66b3ff; font-size: 14px; font-weight: bold;")
        content_layout.addWidget(step1_label)

        step1_text = QLabel("• Выберите тип задачи: максимизация или минимизация\n"
                            "• Укажите формат чисел: десятичные или обыкновенные дроби\n"
                            "• Задайте количество переменных\n"
                            "• Задайте количество ограничений")
        step1_text.setStyleSheet("color: white; padding-left: 10px;")
        step1_text.setWordWrap(True) # перенос на новую строку
        content_layout.addWidget(step1_text)

        step1_img = QLabel()
        pixmap1 = load_image("1.png")
        if not pixmap1.isNull():
            # масштабируем с сохранением пропорций
            scaled_pixmap = pixmap1.scaled(600, 400, 
                                        Qt.AspectRatioMode.KeepAspectRatio, 
                                        Qt.TransformationMode.SmoothTransformation) # пропорции, качество
            step1_img.setPixmap(scaled_pixmap)
            step1_img.setAlignment(Qt.AlignmentFlag.AlignLeft)
            step1_img.setStyleSheet("margin-top: 5px; margin-bottom: 5px;")
            content_layout.addWidget(step1_img)
            content_layout.addSpacing(10)
        else:
            error_label = QLabel("❌ Изображение не найдено: photo/1.png")
            error_label.setStyleSheet("color: #ff6666; font-style: italic;")
            content_layout.addWidget(error_label)

        step2_label = QLabel("2. Заполнение таблиц")
        step2_label.setStyleSheet("color: #66b3ff; font-size: 14px; font-weight: bold;")
        content_layout.addWidget(step2_label)

        step2_text = QLabel("• Нажмите кнопку \"Создать таблицы\"\n"
                            "• Заполните коэффициенты целевой функции\n"
                            "• Заполните коэффициенты ограничений\n"
                            "• Для каждого ограничения выберите тип (≤, ≥, =)\n"
                            "• Введите правую часть ограничений")
        step2_text.setStyleSheet("color: white; padding-left: 10px;")
        step2_text.setWordWrap(True)
        content_layout.addWidget(step2_text)

        step2_img = QLabel()
        pixmap2 = load_image("2.png")
        if not pixmap2.isNull():
            scaled_pixmap = pixmap2.scaled(600, 400,
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
            step2_img.setPixmap(scaled_pixmap)
            step2_img.setAlignment(Qt.AlignmentFlag.AlignLeft)
            step2_img.setStyleSheet("margin-top: 5px; margin-bottom: 5px;")
            content_layout.addWidget(step2_img)
            content_layout.addSpacing(10)
        else:
            error_label = QLabel("❌ Изображение не найдено: photo/2.png")
            error_label.setStyleSheet("color: #ff6666; font-style: italic;")
            content_layout.addWidget(error_label)

        step3_label = QLabel("3. Выбор метода решения")
        step3_label.setStyleSheet("color: #66b3ff; font-size: 14px; font-weight: bold;")
        content_layout.addWidget(step3_label)

        step3_text = QLabel("• \"Решить графически\" - для задач с 2 переменными\n"
                            "• \"Решить симплекс-методом\" - для любого числа переменных")
        step3_text.setStyleSheet("color: white; padding-left: 10px;")
        step3_text.setWordWrap(True)
        content_layout.addWidget(step3_text)

        step3_img = QLabel()
        pixmap3 = load_image("3.png")
        if not pixmap3.isNull():
            scaled_pixmap = pixmap3.scaled(600, 400,
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
            step3_img.setPixmap(scaled_pixmap)
            step3_img.setAlignment(Qt.AlignmentFlag.AlignLeft)
            step3_img.setStyleSheet("margin-top: 5px; margin-bottom: 5px;")
            content_layout.addWidget(step3_img)
            content_layout.addSpacing(10)
        else:
            error_label = QLabel("❌ Изображение не найдено: photo/3.png")
            error_label.setStyleSheet("color: #ff6666; font-style: italic;")
            content_layout.addWidget(error_label)

        step4_label = QLabel("4. Графический метод")
        step4_label.setStyleSheet("color: #66b3ff; font-size: 14px; font-weight: bold;")
        content_layout.addWidget(step4_label)

        step4_text = QLabel("• Область допустимых решений выделяется цветом\n"
                            "• Оптимальная точка помечается красным маркером\n"
                            "• Линия уровня целевой функции отображается пунктиром")
        step4_text.setStyleSheet("color: white; padding-left: 10px;")
        step4_text.setWordWrap(True)
        content_layout.addWidget(step4_text)

        step4_img = QLabel()
        pixmap4 = load_image("4.png")
        if not pixmap4.isNull():
            scaled_pixmap = pixmap4.scaled(600, 400,
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
            step4_img.setPixmap(scaled_pixmap)
            step4_img.setAlignment(Qt.AlignmentFlag.AlignLeft)
            step4_img.setStyleSheet("margin-top: 5px; margin-bottom: 5px;")
            content_layout.addWidget(step4_img)
            content_layout.addSpacing(10)
        else:
            error_label = QLabel("❌ Изображение не найдено: photo/4.png")
            error_label.setStyleSheet("color: #ff6666; font-style: italic;")
            content_layout.addWidget(error_label)

        step5_label = QLabel("5. Симплекс-метод")
        step5_label.setStyleSheet("color: #66b3ff; font-size: 14px; font-weight: bold;")
        content_layout.addWidget(step5_label)

        step5_text = QLabel("• Открывается новое окно с пошаговым решением\n"
                            "• Возможны ручной и автоматический режимы\n"
                            "• Опорные элементы подсвечиваются цветами\n"
                            "• Можно отменять шаги кнопкой \"Назад\"\n"
                            "• Отображается оптимальное решение и значение целевой функции")
        step5_text.setStyleSheet("color: white; padding-left: 10px;")
        step5_text.setWordWrap(True)
        content_layout.addWidget(step5_text)

        step5_img = QLabel()
        pixmap5 = load_image("5.png")
        if not pixmap5.isNull():
            scaled_pixmap = pixmap5.scaled(600, 400,
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
            step5_img.setPixmap(scaled_pixmap)
            step5_img.setAlignment(Qt.AlignmentFlag.AlignLeft)
            step5_img.setStyleSheet("margin-top: 5px; margin-bottom: 5px;")
            content_layout.addWidget(step5_img)
            content_layout.addSpacing(10)
        else:
            error_label = QLabel("❌ Изображение не найдено: photo/5.png")
            error_label.setStyleSheet("color: #ff6666; font-style: italic;")
            content_layout.addWidget(error_label)

        step6_label = QLabel("6. Сохранение и загрузка")
        step6_label.setStyleSheet("color: #66b3ff; font-size: 14px; font-weight: bold;")
        content_layout.addWidget(step6_label)

        step6_text = QLabel("• \"Сохранить в JSON\" - сохраняет задачу в формате JSON\n"
                            "• \"Загрузить из файла\" - загружает задачу из файла")
        step6_text.setStyleSheet("color: white; padding-left: 10px;")
        step6_text.setWordWrap(True)
        content_layout.addWidget(step6_text)

        step6_img = QLabel()
        pixmap6 = load_image("6.png")
        if not pixmap6.isNull():
            scaled_pixmap = pixmap6.scaled(600, 400,
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
            step6_img.setPixmap(scaled_pixmap)
            step6_img.setAlignment(Qt.AlignmentFlag.AlignLeft)
            step6_img.setStyleSheet("margin-top: 5px; margin-bottom: 5px;")
            content_layout.addWidget(step6_img)
            content_layout.addSpacing(10)
        else:
            error_label = QLabel("❌ Изображение не найдено: photo/6.png")
            error_label.setStyleSheet("color: #ff6666; font-style: italic;")
            content_layout.addWidget(error_label)

        step7_label = QLabel("7. Формат файла для загрузки")
        step7_label.setStyleSheet("color: #66b3ff; font-size: 14px; font-weight: bold;")
        content_layout.addWidget(step7_label)

        step7_text = QLabel("• Коэффициенты целевой функции\n"
                            "• Ограничения (коэффициенты, тип знака, правый ограничитель)\n"
                            "• Тип задачи (min, max)")
        step7_text.setStyleSheet("color: white; padding-left: 10px;")
        step7_text.setWordWrap(True)
        content_layout.addWidget(step7_text)

        step7_img = QLabel()
        pixmap7 = load_image("7.png")  
        if not pixmap7.isNull():
            scaled_pixmap = pixmap7.scaled(600, 400,
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
            step7_img.setPixmap(scaled_pixmap)
            step7_img.setAlignment(Qt.AlignmentFlag.AlignLeft)
            step7_img.setStyleSheet("margin-top: 5px; margin-bottom: 5px;")
            content_layout.addWidget(step7_img)
            content_layout.addSpacing(10)
        else:
            error_label = QLabel("❌ Изображение не найдено: photo/7.png")
            error_label.setStyleSheet("color: #ff6666; font-style: italic;")
            content_layout.addWidget(error_label)

        content_layout.addSpacing(20)
        author_label = QLabel("Лепехина Алена, ИВТ-31БО")
        author_label.setStyleSheet("color: #88ccff; font-weight: bold; font-style: italic;")
        author_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        content_layout.addWidget(author_label)

        # растягивающийся спейсер в конец
        content_layout.addStretch(1)

        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

    # открытие окна симплекс-метода
    def open_simplex_window(self):
        try:
            num_vars = self.vars_spin.value()
            num_constraints = self.constraints_spin.value()

            # собираем коэффициенты целевой функции из таблицы
            basic_func = []
            for i in range(num_vars):
                item = self.objective_table.item(0, i)
                basic_func.append(Fraction(item.text()))

            # ограничения
            constraints = []
            for row in range(num_constraints):
                constraint_coeffs = []
                for col in range(num_vars):
                    item = self.constraints_table.item(row, col)
                    constraint_coeffs.append(Fraction(item.text()))
                
                # получаем правую часть ограничения
                rhs_item = self.constraints_table.item(row, num_vars + 1)
                rhs = Fraction(rhs_item.text())

                constraints.append(
                    {
                        "type": "=",
                        "coeff": constraint_coeffs,
                        "value": rhs,  
                    }
                )

            minimize = self.problem_type_combo.currentText() == "Минимизация"
            use_fractions = (
                self.format_combo.currentText() == "Обыкновенные дроби"
            )

            simplex_win = SimplexWindow(
                parent=self,
                basic_func=basic_func,
                constraints=constraints,
                minimize=minimize,
                use_fractions=use_fractions,
            )
            simplex_win.exec() # блокируем родительское окно 

        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Ошибка при запуске симплекс-метода:\n{str(e)}",
            )

    def update_button_styles(self):
        additional_styles = """
            QPushButton.important {
                background-color: rgb(0, 85, 127);
                border: 1px solid rgb(0, 105, 157);
            }
            /* Стиль при наведении курсора на важные кнопки */
            QPushButton.important:hover {
                background-color: rgb(0, 95, 142);
                border: 1px solid rgb(0, 115, 172);
            }
            /* Стиль при нажатии на важные кнопки */
            QPushButton.important:pressed {
                background-color: rgb(0, 75, 112);
                border: 1px solid rgb(0, 85, 127);
            }
        """
        self.setStyleSheet(self.styleSheet() + additional_styles)

    # создание таблиц ввода на основе выбранных параметров
    def create_tables(self):
        num_vars = self.vars_spin.value()
        num_constraints = self.constraints_spin.value()

        # целевая функция 
        self.objective_table.clearContents()
        self.objective_table.setRowCount(0)
        self.objective_table.setColumnCount(0)

        self.objective_table.setColumnCount(num_vars)
        self.objective_table.setRowCount(1)

        for i in range(num_vars):
            coeff_item = QTableWidgetItem("0")
            self.objective_table.setItem(0, i, coeff_item)

        # ограничения 
        self.constraints_table.clearContents()
        self.constraints_table.setRowCount(0)
        self.constraints_table.setColumnCount(0)

        self.constraints_table.setColumnCount(num_vars + 2) # +1 для типа ограничения, +1 для правой части
        self.constraints_table.setRowCount(num_constraints)

        for row in range(num_constraints):
            for col in range(num_vars):
                coeff_item = QTableWidgetItem("0")
                self.constraints_table.setItem(row, col, coeff_item)

            # столбец для типа ограничения
            type_combo = QComboBox()
            type_combo.addItems(["=", "≤", "≥"])
            self.constraints_table.setCellWidget(row, num_vars, type_combo)

            # столбец для правой части
            rhs_item = QTableWidgetItem("0")
            self.constraints_table.setItem(row, num_vars + 1, rhs_item)

    # решение задачи графическим методом
    def solve_problem(self):
        try:
            # получаем данные из таблиц
            num_vars = self.vars_spin.value()
            num_constraints = self.constraints_spin.value()

            if num_vars != 2:
                QMessageBox.warning(
                    self,
                    "Ошибка",
                    "Графический метод работает только с 2 переменными!",
                )
                return

            if self.format_combo.currentText() == "Обыкновенные дроби":
                self.format = Fraction
            else:
                self.format = float
            # целевая функция
            c = []
            for i in range(num_vars):
                item = self.objective_table.item(0, i)
                c.append(self.format(item.text()))

            # ограничения
            constraints = []
            for row in range(num_constraints):
                constraint_coeffs = []
                for col in range(num_vars):
                    item = self.constraints_table.item(row, col)
                    constraint_coeffs.append(self.format(item.text()))

                # тип ограничения
                combo = self.constraints_table.cellWidget(row, num_vars)
                constraint_type = combo.currentText()
                # преобразуем в формат для graphical_method
                if constraint_type == "≤":
                    constr_type = "<="
                elif constraint_type == "≥":
                    constr_type = ">="
                else:
                    constr_type = "="

                # правая часть
                rhs_item = self.constraints_table.item(row, num_vars + 1)
                rhs = self.format(rhs_item.text())

                constraints.append(
                    {
                        "type": constr_type,
                        "coeff": constraint_coeffs,
                        "value": rhs,
                    }
                )

            # границы переменных 
            bounds = {"x₁": (0, None), "x₂": (0, None)}

            # тип задачи
            minimize = self.problem_type_combo.currentText() == "Минимизация"

            # вызываем графический метод
            solution, z = self.graphical_method(
                c, constraints, bounds, minimize
            )

            # выводим результаты
            if solution:
                result_text = f"Оптимальное решение: x₁ = {solution[0]:.2f}, x₂ = {solution[1]:.2f}\n"
                result_text += f"Значение целевой функции: {'min' if minimize else 'max'} = {z:.2f}\n\n"

                result_text += "Целевая функция:\n"
                result_text += (
                    f"{c[0]}x₁ + {c[1]}x₂ → {'min' if minimize else 'max'}\n\n"
                )

                result_text += "Ограничения:\n"
                for i, constr in enumerate(constraints):
                    result_text += f"{constr['coeff'][0]}x₁ + {constr['coeff'][1]}x₂ {constr['type']} {constr['value']}\n"

                self.result_output.setText(result_text)
                self.tabs.setCurrentIndex(
                    1
                )  # переключаемся на вкладку с результатами
            else:
                self.result_output.setText("Допустимая область пуста!")
                self.tabs.setCurrentIndex(1)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Произошла ошибка при решении задачи:\n{str(e)}",
            )


    # метод для графического решения задачи
    def graphical_method(self, c, constraints, bounds, minimize=True):
        global feasible_side # для хранения допустимой стороны
        self.canvas.axes.xaxis.label.set_color("white")
        self.canvas.axes.yaxis.label.set_color("white")
        self.canvas.axes.title.set_color("white")
        self.canvas.axes.tick_params(colors="white", which="both")
        self.canvas.axes.spines["bottom"].set_color("white")
        self.canvas.axes.spines["top"].set_color("white")
        self.canvas.axes.spines["right"].set_color("white")
        self.canvas.axes.spines["left"].set_color("white")
        self.canvas.fig.set_facecolor("#2D2D2D")
        self.canvas.axes.set_facecolor("#2D2D2D")
        self.canvas.axes.clear() # очищаем предыдущий график

        # обработка ограничений и построение линий
        x1_vals = np.linspace(0, 10, 400) # диапазон значений x₁ от 0 до 10 (400 точек)
        feasible_polygons = []

        for constraint in constraints:
            a, b = constraint["coeff"]
            c_val = constraint["value"]

            if b != 0:
                x2_vals = (c_val - a * x1_vals) / b
            else:
                x2_vals = np.full_like(
                    x1_vals, c_val / a
                )  # вертикальная линия x = c/a

            line = LineString(np.column_stack((x1_vals, x2_vals)))

            # определение допустимой полуплоскости
            if constraint["type"] == "<=":
                if b > 0: # # для a·x₁ + b·x₂ ≤ c, если b > 0, то ниже линии
                    feasible_side = np.column_stack((x1_vals, x2_vals - 1e5))
                else: #  # если b < 0, то выше линии
                    feasible_side = np.column_stack((x1_vals, x2_vals + 1e5))
            elif constraint["type"] == ">=":
                if b > 0:
                    feasible_side = np.column_stack((x1_vals, x2_vals + 1e5))
                else:
                    feasible_side = np.column_stack((x1_vals, x2_vals - 1e5))

             # создаем полигон допустимой полуплоскости
            feasible_poly = (
                Polygon(line).union(Polygon(feasible_side)).convex_hull # объединяем, ищем выпукл.обл.
            )
            feasible_polygons.append(feasible_poly)

            # построение линии ограничения
            self.canvas.axes.plot(
                x1_vals,
                x2_vals,
                label=f"{a}x₁ + {b}x₂ {constraint['type']} {c_val}",
            )

        # нахождение допустимой области
        feasible_region = feasible_polygons[0]
        for poly in feasible_polygons[1:]:
            feasible_region = feasible_region.intersection(poly) # поиск общего

        if feasible_region.is_empty:
            self.canvas.axes.set_title("Допустимая область пуста!")
            self.canvas.draw()
            return None, None

        # визуализация допустимой области
        if isinstance(feasible_region, Polygon):
            x, y = feasible_region.exterior.xy # для определения положения точки в пространстве
            self.canvas.axes.fill(
                x, y, alpha=0.2, color="gray", label="Допустимая область"
            )
        else:
            # если область состоит из нескольких полигонов
            for geom in feasible_region.geoms:
                x, y = geom.exterior.xy
                self.canvas.axes.fill(x, y, alpha=0.2, color="gray")

        # поиск угловых точек
        if isinstance(feasible_region, Polygon):
            vertices = list(feasible_region.exterior.coords)
        else:
            vertices = []
            for geom in feasible_region.geoms:
                vertices.extend(list(geom.exterior.coords))

        # удаление дубликатов
        vertices = list(set(vertices))

        # вычисление значений целевой функции в угловых точках
        z_values = [c[0] * x + c[1] * y for x, y in vertices]

        if minimize:
            opt_idx = np.argmin(z_values)
            opt_type = "минимум"
        else:
            opt_idx = np.argmax(z_values)
            opt_type = "максимум"

        opt_x, opt_y = vertices[opt_idx]
        opt_z = z_values[opt_idx]

        # отметка оптимальной точки
        self.canvas.axes.scatter(
            opt_x,
            opt_y,
            color="red",
            s=100,
            label=f"Оптимум ({opt_type}): ({opt_x:.2f}, {opt_y:.2f})",
        )

        # построение линий уровня целевой функции
        if c[1] != 0:
            level_y = (opt_z - c[0] * x1_vals) / c[1]
            self.canvas.axes.plot(
                x1_vals,
                level_y,
                "--",
                color="green",
                label=f"Целевая: {c[0]}x₁ + {c[1]}x₂ = {opt_z:.2f}",
            )
        else:
            self.canvas.axes.axvline(
                x=opt_z / c[0],
                linestyle="--",
                color="green",
                label=f"Целевая: {c[0]}x₁ = {opt_z:.2f}",
            )

        # настройка графика
        self.canvas.axes.set_xlabel("x₁")
        self.canvas.axes.set_ylabel("x₂")
        self.canvas.axes.set_xlim(0, max(10, opt_x * 1.2))
        self.canvas.axes.set_ylim(0, max(10, opt_y * 1.2))
        self.canvas.axes.legend()
        self.canvas.axes.grid(True)
        self.canvas.axes.set_title(
            "Графический метод линейного программирования"
        )
        self.canvas.draw()

        return (opt_x, opt_y), opt_z

    def load_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть файл задачи",
            "",
            "JSON файлы (*.json);;Текстовые файлы (*.txt);;Все файлы (*)",
        )

        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.endswith(".json"):
                    data = json.load(f)
                    self.load_from_json(data)
                else:
                    lines = [
                        line.strip() for line in f.readlines() if line.strip()
                    ]
                    self.load_from_text(lines)

            QMessageBox.information(self, "Успех", "Задача успешно загружена!")

        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка", f"Ошибка при чтении файла:\n{str(e)}"
            )

    def load_from_json(self, data):
        function = data.get("function", [])
        constraints_data = data.get("constraints", [])
        minmax = data.get("minmax", "max")

        num_vars = len(function)
        num_constraints = len(constraints_data)

        if num_vars == 0 or num_constraints == 0:
            raise ValueError(
                "Некорректный JSON: пустая функция или ограничения"
            )

        # устанавливаем тип задачи
        self.problem_type_combo.setCurrentText(
            "Минимизация" if minmax == "min" else "Максимизация"
        )

        # устанавливаем размерности и создаём таблицы
        self.vars_spin.setValue(num_vars)
        self.constraints_spin.setValue(num_constraints)
        self.create_tables()

        # заполняем целевую функцию
        for i, coeff in enumerate(function):
            self.objective_table.setItem(0, i, QTableWidgetItem(str(coeff)))

        # заполняем ограничения
        for row, constr in enumerate(constraints_data):
            coeffs = constr.get("coeffs", [])
            constr_type = constr.get("type", "<=")
            rhs = constr.get("rhs", "0")

            if len(coeffs) != num_vars:
                raise ValueError(
                    f"Несоответствие числа переменных в ограничении {row + 1}"
                )

            # коэффициенты
            for col in range(num_vars):
                self.constraints_table.setItem(
                    row, col, QTableWidgetItem(str(coeffs[col]))
                )

            # тип ограничения
            combo = self.constraints_table.cellWidget(row, num_vars)
            if constr_type == "≤":
                combo.setCurrentIndex(1)
            elif constr_type == "≥":
                combo.setCurrentIndex(2)
            elif constr_type == "=":
                combo.setCurrentIndex(0)
            else:
                # на случай неожиданных значений — по умолчанию "≤"
                combo.setCurrentIndex(0)

            # правая часть
            self.constraints_table.setItem(
                row, num_vars + 1, QTableWidgetItem(str(rhs))
            )


    def load_from_text(self, lines):
        try:
            # Первая строка: количество переменных и ограничений
            first_line = lines[0].split()
            if len(first_line) < 2:
                raise ValueError("Первая строка должна содержать количество переменных и ограничений")
            
            num_vars = int(first_line[0])
            num_constraints = int(first_line[1])
            
            # Вторая строка: целевая функция
            if len(lines) < 2:
                raise ValueError("Отсутствует строка с целевой функцией")
            
            objective_coeffs = lines[1].split()
            if len(objective_coeffs) != num_vars:
                raise ValueError(f"Ожидалось {num_vars} коэффициентов в целевой функции, получено {len(objective_coeffs)}")
            
            # Ограничения
            constraints = []
            constraint_lines = lines[2:2 + num_constraints]
            
            for i, line in enumerate(constraint_lines):
                parts = line.split()
                if len(parts) != num_vars + 2:
                    raise ValueError(f"Некорректный формат ограничения {i+1}: ожидалось {num_vars+2} элементов")
                
                coeffs = parts[:num_vars]
                constr_type = parts[num_vars]  # =, ≤, ≥
                rhs = parts[num_vars + 1]
                
                constraints.append({
                    "coeffs": coeffs,
                    "type": constr_type,
                    "rhs": rhs
                })
            
            # Определяем тип задачи (последняя строка, если есть)
            minmax = "max"  # по умолчанию максимизация
            if len(lines) > 2 + num_constraints:
                last_line = lines[2 + num_constraints].strip().lower()
                if last_line in ["min", "max"]:
                    minmax = last_line
            
            # Устанавливаем значения в интерфейсе
            self.problem_type_combo.setCurrentText(
                "Минимизация" if minmax == "min" else "Максимизация"
            )
            
            self.vars_spin.setValue(num_vars)
            self.constraints_spin.setValue(num_constraints)
            self.create_tables()
            
            # Заполняем целевую функцию
            for i, coeff in enumerate(objective_coeffs):
                self.objective_table.setItem(0, i, QTableWidgetItem(coeff))
            
            # Заполняем ограничения
            for row, constr in enumerate(constraints):
                coeffs = constr["coeffs"]
                constr_type = constr["type"]
                rhs = constr["rhs"]
                
                for col in range(num_vars):
                    self.constraints_table.setItem(row, col, QTableWidgetItem(coeffs[col]))
                
                combo = self.constraints_table.cellWidget(row, num_vars)
                if constr_type == "≤":
                    combo.setCurrentIndex(1)
                elif constr_type == "≥":
                    combo.setCurrentIndex(2)
                elif constr_type == "=":
                    combo.setCurrentIndex(0)
                else:
                    combo.setCurrentIndex(0)
                
                self.constraints_table.setItem(row, num_vars + 1, QTableWidgetItem(rhs))
                
        except Exception as e:
            raise ValueError(f"Ошибка при чтении текстового файла: {str(e)}")


    def save_to_json(self):
        try:
            num_vars = self.vars_spin.value()
            num_constraints = self.constraints_spin.value()

            # целевая функция
            function = []
            for i in range(num_vars):
                item = self.objective_table.item(0, i)
                val = (
                    item.text().strip()
                    if item and item.text().strip()
                    else "0"
                )
                function.append(val)

            # сбор ограничений с типами
            constraints = []
            for row in range(num_constraints):
                # коэффициенты
                coeffs = []
                for col in range(num_vars):
                    item = self.constraints_table.item(row, col)
                    val = (
                        item.text().strip()
                        if item and item.text().strip()
                        else "0"
                    )
                    coeffs.append(val)

                # тип ограничения
                combo = self.constraints_table.cellWidget(row, num_vars)
                type_str = combo.currentText()  # "≤", "≥", или "="

                # правая часть
                rhs_item = self.constraints_table.item(row, num_vars + 1)
                rhs = (
                    rhs_item.text().strip()
                    if rhs_item and rhs_item.text().strip()
                    else "0"
                )

                constraints.append(
                    {"coeffs": coeffs, "type": type_str, "rhs": rhs}
                )

            # тип задачи
            minmax = (
                "min"
                if self.problem_type_combo.currentText() == "Минимизация"
                else "max"
            )

            data = {
                "function": function,
                "constraints": constraints,
                "minmax": minmax,
            }

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить задачу",
                "",
                "JSON файлы (*.json);;Все файлы (*)",
            )
            if file_path:
                if not file_path.endswith(".json"):
                    file_path += ".json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                QMessageBox.information(
                    self, "Успех", "Задача успешно сохранена в JSON!"
                )

        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка", f"Не удалось сохранить задачу:\n{str(e)}"
            )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LinearProgrammingApp()
    window.show()
    sys.exit(app.exec())
