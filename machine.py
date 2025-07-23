import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import pandas as pd
import xgboost as xgb
from urllib.parse import urlparse
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class PhishingEDAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализ фишинговых сайтов")
        self.root.geometry("1200x800")
        self.df = None
        self.filtered_df = None
        self.model = None
        self.scaler = StandardScaler()
        self.used_features = None

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        self.header_frame = ctk.CTkFrame(self.root, corner_radius=0)
        self.header_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.header_frame.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.header_frame, 
            text="ФИШИНГ-недоМАСТЕР",
            font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=20, pady=10)

        self.main_container = ctk.CTkFrame(self.root, corner_radius=0)
        self.main_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.main_container.grid_columnconfigure(0, weight=0)
        self.main_container.grid_columnconfigure(1, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self.main_container, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="ns", padx=(5, 0), pady=5)
        self.sidebar.grid_rowconfigure(8, weight=1)

        self.content = ctk.CTkFrame(self.main_container, corner_radius=0)
        self.content.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        self.load_btn = ctk.CTkButton(
            self.sidebar, 
            text="Загрузить CSV", 
            command=self.load_csv,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"))
        self.load_btn.grid(row=0, column=0, padx=20, pady=(10, 5), sticky="ew")

        self.info_label = ctk.CTkLabel(
            self.sidebar, 
            text="CSV не загружен",
            wraplength=200,
            justify="left",
            font=ctk.CTkFont(size=12))
        self.info_label.grid(row=1, column=0, padx=20, pady=5, sticky="w")

        self.https_label = ctk.CTkLabel(
            self.sidebar, 
            text="HTTPS:",
            font=ctk.CTkFont(size=12))
        self.https_label.grid(row=3, column=0, padx=20, pady=(5, 2), sticky="w")
        
        self.https_var = ctk.StringVar(value="Все")
        self.https_combo = ctk.CTkComboBox(
            self.sidebar,
            variable=self.https_var,
            values=["Все", "Есть (0)", "Нет (1)"],
            state="readonly",
            width=200)
        self.https_combo.grid(row=4, column=0, padx=20, pady=(2, 5))

        self.ip_label = ctk.CTkLabel(
            self.sidebar, 
            text="IP-адрес:",
            font=ctk.CTkFont(size=12))
        self.ip_label.grid(row=5, column=0, padx=20, pady=(5, 2), sticky="w")
        
        self.ip_var = ctk.StringVar(value="Все")
        self.ip_combo = ctk.CTkComboBox(
            self.sidebar,
            variable=self.ip_var,
            values=["Все", "Домен (0)", "IP (1)"],
            state="readonly",
            width=200)
        self.ip_combo.grid(row=6, column=0, padx=20, pady=(2, 5))

        self.class_label = ctk.CTkLabel(
            self.sidebar, 
            text="Класс:",
            font=ctk.CTkFont(size=12))
        self.class_label.grid(row=7, column=0, padx=20, pady=(5, 2), sticky="w")
        
        self.class_var = ctk.StringVar(value="Все")
        self.class_combo = ctk.CTkComboBox(
            self.sidebar,
            variable=self.class_var,
            values=["Все", "Легитимные (0)", "Фишинговые (1)"],
            state="readonly",
            width=200)
        self.class_combo.grid(row=8, column=0, padx=20, pady=(2, 10))

        self.filter_btn = ctk.CTkButton(
            self.sidebar, 
            text="Применить фильтры", 
            command=self.apply_filters,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"))
        self.filter_btn.grid(row=9, column=0, padx=20, pady=5, sticky="ew")

        self.recommend_btn = ctk.CTkButton(
            self.sidebar, 
            text="Показать выводы", 
            command=self.show_recommendations,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"))
        self.recommend_btn.grid(row=10, column=0, padx=20, pady=5, sticky="ew")

        self.train_btn = ctk.CTkButton(
            self.sidebar, 
            text="Обучить модель", 
            command=self.train_model,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"))
        self.train_btn.grid(row=11, column=0, padx=20, pady=5, sticky="ew")

        self.appearance_mode = ctk.StringVar(value="System")
        self.appearance_label = ctk.CTkLabel(
            self.sidebar, 
            text="Тема:",
            font=ctk.CTkFont(size=12))
        self.appearance_label.grid(row=12, column=0, padx=20, pady=(5, 2), sticky="w")
        
        self.appearance_combo = ctk.CTkComboBox(
            self.sidebar,
            variable=self.appearance_mode,
            values=["System", "Light", "Dark"],
            state="readonly",
            width=200,
            command=self.change_appearance_mode)
        self.appearance_combo.grid(row=13, column=0, padx=20, pady=(2, 10))

        self.tabview = ctk.CTkTabview(self.content)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.tabview.add("Датасет")
        self.tabview.add("Статистика")
        self.tabview.add("Предсказание")

        for tab in ["Датасет", "Статистика", "Предсказание"]:
            self.tabview.tab(tab).grid_columnconfigure(0, weight=1)
            self.tabview.tab(tab).grid_rowconfigure(0, weight=1)

        self.tree_frame = ctk.CTkFrame(self.tabview.tab("Датасет"))
        self.tree_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.tree_frame.grid_columnconfigure(0, weight=1)
        self.tree_frame.grid_rowconfigure(0, weight=1)

        self.tree_scroll = ctk.CTkScrollableFrame(self.tree_frame)
        self.tree_scroll.grid(row=0, column=0, sticky="nsew")
        self.tree_scroll.grid_columnconfigure(0, weight=1)
        self.tree_scroll.grid_rowconfigure(0, weight=1)
        self.tree = None

        self.stats_frame = ctk.CTkFrame(self.tabview.tab("Статистика"))
        self.stats_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.stats_frame.grid_columnconfigure(0, weight=1)
        self.stats_frame.grid_rowconfigure(0, weight=1)

        self.stats_scroll = ctk.CTkScrollableFrame(self.stats_frame)
        self.stats_scroll.grid(row=0, column=0, sticky="nsew")
        self.stats_scroll.grid_columnconfigure(0, weight=1)
        self.stats_scroll.grid_rowconfigure(0, weight=1)
        self.stats_tree = None

        self.predict_frame = ctk.CTkFrame(self.tabview.tab("Предсказание"))
        self.predict_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.predict_frame.grid_columnconfigure(0, weight=1)
        self.predict_frame.grid_rowconfigure(2, weight=1)

        self.url_label = ctk.CTkLabel(
            self.predict_frame,
            text="Введите URL для проверки:",
            font=ctk.CTkFont(size=14))
        self.url_label.grid(row=0, column=0, padx=20, pady=(10, 5), sticky="w")
        
        self.url_entry = ctk.CTkEntry(
            self.predict_frame,
            width=400,
            placeholder_text="https://example.com")
        self.url_entry.grid(row=1, column=0, padx=20, pady=5, sticky="w")

        self.predict_btn = ctk.CTkButton(
            self.predict_frame,
            text="Проверить URL",
            command=self.predict_url,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"))
        self.predict_btn.grid(row=1, column=1, padx=20, pady=5)

        self.result_text = ctk.CTkTextbox(
            self.predict_frame,
            height=200,
            wrap="word")
        self.result_text.grid(row=2, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")

        self.root.bind("<Configure>", self.on_resize)
    
    def on_resize(self, event):
        window_width = self.root.winfo_width()
        sidebar_width = min(max(200, window_width // 4), 300)
        self.sidebar.configure(width=sidebar_width)
        self.info_label.configure(wraplength=sidebar_width - 40)
        self.https_combo.configure(width=sidebar_width - 40)
        self.ip_combo.configure(width=sidebar_width - 40)
        self.class_combo.configure(width=sidebar_width - 40)
    
    def change_appearance_mode(self, new_mode):
        ctk.set_appearance_mode(new_mode)
    
    def extract_url_features(self, url):
        parsed_url = urlparse(url)
        features = {
            'no_https': 1 if parsed_url.scheme != 'https' else 0,
            'ip_address': 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', parsed_url.netloc) else 0,
            'url_length': len(url),
            'num_digits': sum(c.isdigit() for c in url),
            'num_special_chars': sum(not c.isalnum() for c in url),
            'pct_ext_hyperlinks': 0.0,
        }
        return pd.DataFrame([features])
    
    def train_model(self):
        if self.df is None:
            messagebox.showwarning("Внимание", "Сначала CSV потом балуйся")
            return
        
        try:
            possible_features = ['no_https', 'ip_address', 'url_length', 'num_digits', 
                               'num_special_chars', 'pct_ext_hyperlinks']
            if 'c_l_a_s_s__l_a_b_e_l' not in self.df.columns:
                messagebox.showerror("Ошибка", "Отсутствует столбец с метками: c_l_a_s_s__l_a_b_e_l")
                return
            available_features = [col for col in possible_features if col in self.df.columns]
            if not available_features:
                messagebox.showerror("Ошибка", "Нет доступных признаков для обучения модели")
                return

            X = self.df[available_features]
            y = self.df['c_l_a_s_s__l_a_b_e_l']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state =42)

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)

            self.used_features = available_features

            y_pred = self.model.predict(X_test_scaled)
            report = classification_report(y_test, y_pred)

            messagebox.showinfo("Леееес гоу", f"Модель обучена!\nИспользованы признаки: {', '.join(available_features)}\n\nМетрики:\n{report}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при обучении модели: {e}")
    
    def predict_url(self):
        if self.model is None:
            messagebox.showwarning("Внимание", "Сначала обучите модель")
            return
        
        url = self.url_entry.get()
        if not url:
            messagebox.showwarning("Внимание", "Введите URL")
            return
        
        try:
            parsed_url = urlparse(url)
            features = {
                'no_https': 1 if parsed_url.scheme != 'https' else 0,
                'ip_address': 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', parsed_url.netloc) else 0,
                'url_length': len(url),
                'num_digits': sum(c.isdigit() for c in url),
                'num_special_chars': sum(not c.isalnum() for c in url),
                'pct_ext_hyperlinks': 0.0,
            }

            used_features = {k: v for k, v in features.items() if k in self.used_features}
            features_df = pd.DataFrame([used_features])

            if set(self.used_features) != set(used_features.keys()):
                missing = set(self.used_features) - set(used_features.keys())
                messagebox.showerror("Ошибка", f"Отсутствуют признаки для предсказания: {', '.join(missing)}")
                return

            features_scaled = self.scaler.transform(features_df)
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]

            result = (f"URL: {url}\n\n"
                     f"Предсказание: {'Фишинговый' if prediction == 1 else 'Легитимный'}\n"
                     f"Вероятность фишинга: {probability[1]:.2%}\n"
                     f"Вероятность легитимности: {probability[0]:.2%}\n\n"
                     f"Признаки:\n")
            for feature, value in used_features.items():
                if feature == 'no_https':
                    result += f"- HTTPS: {'Отсутствует' if value == 1 else 'Присутствует'}\n"
                elif feature == 'ip_address':
                    result += f"- IP-адрес: {'Используется' if value == 1 else 'Домен'}\n"
                else:
                    result += f"- {feature}: {value}\n"
            
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert("1.0", result)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при анализе URL: {e}")
    
    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV файлы", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.filtered_df = self.df.copy()

                info = (f"Данные загружены:\n"
                        f"Строк: {self.df.shape[0]}\n"
                        f"Столбцов: {self.df.shape[1]}\n"
                        f"Размер: {self.df.memory_usage().sum() / 1024:.2f} KB")
                self.info_label.configure(text=info)

                self.create_treeview()
                self.update_statistics()
                
                messagebox.showinfo("Ура", "Данные успешно загружены!")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {e}")
    
    def create_treeview(self):
        if self.tree is not None:
            self.tree.destroy()

        tree_container = ctk.CTkFrame(self.tree_scroll)
        tree_container.pack(fill="both", expand=True)
        tree_container.grid_columnconfigure(0, weight=1)
        tree_container.grid_rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(tree_container, show="headings")

        self.tree["columns"] = list(self.df.columns)
        for col in self.df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, minwidth=80, stretch=tk.YES)

        sample_df = self.df.head(200)
        for _, row in sample_df.iterrows():
            self.tree.insert("", "end", values=list(row))

        y_scroll = ttk.Scrollbar(tree_container, orient="vertical", command=self.tree.yview)
        x_scroll = ttk.Scrollbar(tree_container, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        tree_container.grid_columnconfigure(0, weight=1)
        tree_container.grid_rowconfigure(0, weight=1)
    
    def update_statistics(self):
        if self.df is None:
            return
        
        if self.stats_tree is not None:
            self.stats_tree.destroy()

        stats_container = ctk.CTkFrame(self.stats_scroll)
        stats_container.pack(fill="both", expand=True)
        stats_container.grid_columnconfigure(0, weight=1)
        stats_container.grid_rowconfigure(0, weight=1)

        self.stats_tree = ttk.Treeview(stats_container, show="headings")

        self.stats_tree["columns"] = ["Статистика", "Значение"]
        self.stats_tree.heading("Статистика", text="Статистика")
        self.stats_tree.heading("Значение", text="Значение")
        self.stats_tree.column("Статистика", width=200, minwidth=150, stretch=tk.YES)
        self.stats_tree.column("Значение", width=200, minwidth=150, stretch=tk.YES)
        self.stats_tree.insert("", "end", values=["Записей", str(self.df.shape[0])])
        self.stats_tree.insert("", "end", values=["Столбцов", str(self.df.shape[1])])
        self.stats_tree.insert("", "end", values=["Объем (KB)", f"{self.df.memory_usage().sum() / 1024:.2f}"])

        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            self.stats_tree.insert("", "end", values=["", ""])
            self.stats_tree.insert("", "end", values=["Числовые столбцы", ""])
            desc = self.df[numeric_cols].describe()
            for col in numeric_cols:
                for stat in desc.index:
                    value = f"{desc.loc[stat, col]:.2f}"
                    self.stats_tree.insert("", "end", values=[f"{col} - {stat}", value])

        cat_cols = self.df.select_dtypes(include=['object']).columns
        if not cat_cols.empty:
            self.stats_tree.insert("", "end", values=["", ""])
            self.stats_tree.insert("", "end", values=["Категориальные столбцы", ""])
            for col in cat_cols:
                value_counts = self.df[col].value_counts()
                for value, count in value_counts.items():
                    self.stats_tree.insert("", "end", values=[f"{col} - {value}", str(count)])

        y_scroll = ttk.Scrollbar(stats_container, orient="vertical", command=self.stats_tree.yview)
        x_scroll = ttk.Scrollbar(stats_container, orient="horizontal", command=self.stats_tree.xview)
        self.stats_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        self.stats_tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        stats_container.grid_columnconfigure(0, weight=1)
        stats_container.grid_rowconfigure(0, weight=1)
    
    def apply_filters(self):
        if self.df is None:
            messagebox.showwarning("Внимание", "Сначала загрузите данные")
            return
        
        try:
            self.filtered_df = self.df.copy()
            if self.https_var.get() != "Все" and 'no_https' in self.df.columns:
                value = int(self.https_var.get()[-2])
                self.filtered_df = self.filtered_df[self.filtered_df["no_https"] == value]
            if self.ip_var.get() != "Все" and 'ip_address' in self.df.columns:
                value = int(self.ip_var.get()[-2])
                self.filtered_df = self.filtered_df[self.filtered_df["ip_address"] == value]
            if self.class_var.get() != "Все" and 'c_l_a_s_s__l_a_b_e_l' in self.df.columns:
                value = int(self.class_var.get()[-2])
                self.filtered_df = self.filtered_df[self.filtered_df["c_l_a_s_s__l_a_b_e_l"] == value]
            
            self.update_treeview()
            self.update_statistics()
            
            messagebox.showinfo("Фильтры применены", 
                              f"Отфильтровано записей: {len(self.filtered_df)}\n"
                              f"(изначально: {len(self.df)})")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при применении фильтров: {e}")
    
    def update_treeview(self):
        if self.tree is None or self.filtered_df is None:
            return
        
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        sample_df = self.filtered_df.head(200)
        for _, row in sample_df.iterrows():
            self.tree.insert("", "end", values=list(row))
    
    def show_recommendations(self):
        if self.df is None:
            messagebox.showwarning("Внимание", "Сначала загрузите данные")
            return
        
        rec_window = ctk.CTkToplevel(self.root)
        rec_window.title("Аналитические выводы и рекомендации")
        rec_window.geometry("800x600")
        rec_window.transient(self.root)
        rec_window.grab_set()
        
        rec_window.grid_columnconfigure(0, weight=1)
        rec_window.grid_rowconfigure(0, weight=1)
        
        tabview = ctk.CTkTabview(rec_window)
        tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        tabview.add("Ключевые выводы")
        tabview.add("Рекомендации")
        tabview.add("Следующие шаги")
        
        for tab in ["Ключевые выводы", "Рекомендации", "Следующие шаги"]:
            tabview.tab(tab).grid_columnconfigure(0, weight=1)
            tabview.tab(tab).grid_rowconfigure(0, weight=1)
        
        try:
            findings_text = "Ключевые выводы\n\n"
            if 'no_https' in self.df.columns:
                findings_text += (
                    "1. Использование HTTPS:\n"
                    f"   - Сайты без HTTPS (no_https=1) чаще являются фишинговыми.\n"
                    f"   - {self.df[self.df['c_l_a_s_s__l_a_b_e_l'] == 1]['no_https'].mean():.1%} "
                    f"фишинговых сайтов не используют HTTPS против "
                    f"{self.df[self.df['c_l_a_s_s__l_a_b_e_l'] == 0]['no_https'].mean():.1%} легитимных.\n\n"
                )
            if 'ip_address' in self.df.columns:
                findings_text += (
                    "2. Использование IP-адресов:\n"
                    f"   - Использование IP вместо домена (ip_address=1) характерно для фишинга.\n"
                    f"   - {self.df[self.df['c_l_a_s_s__l_a_b_e_l'] == 1]['ip_address'].mean():.1%} "
                    f"фишинговых сайтов используют IP против "
                    f"{self.df[self.df['c_l_a_s_s__l_a_b_e_l'] == 0]['ip_address'].mean():.1%} легитимных.\n\n"
                )
            if 'pct_ext_hyperlinks' in self.df.columns:
                findings_text += (
                    "3. Внешние ссылки:\n"
                    f"   - Фишинговые сайты содержат больше внешних ссылок.\n"
                    f"   - В среднем: {self.df[self.df['c_l_a_s_s__l_a_b_e_l'] == 1]['pct_ext_hyperlinks'].mean():.1f} "
                    f"для фишинга против {self.df[self.df['c_l_a_s_s__l_a_b_e_l'] == 0]['pct_ext_hyperlinks'].mean():.1f} "
                    f"для легитимных сайтов.\n"
                )
            if findings_text == "Ключевые выводы\n\n":
                findings_text += "Нет доступных данных для анализа.\n"
        except Exception as e:
            findings_text = f"Ошибка при вычислении выводов: {e}"
        
        findings_tab = ctk.CTkTextbox(tabview.tab("Ключевые выводы"), wrap="word")
        findings_tab.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        findings_tab.insert("1.0", findings_text)
        findings_tab.configure(state="disabled")
        
        rec_text = (
            "Рекомендации\n\n"
            "1. Улучшение безопасности:\n"
            "   - Внедрить проверку HTTPS и паттернов IP/домена в реальном времени.\n"
            "   - Добавить эти функции в существующие системы безопасности.\n\n"
            "2. Обучение пользователей:\n"
            "   - Обучить сотрудников распознаванию подозрительных URL.\n"
            "   - Акцент на наличии HTTPS и структуре доменных имен.\n\n"
            "3. Технические решения:\n"
            "   - Разработать плагины для браузеров с анализом URL.\n"
            "   - Создать систему мониторинга доменов организации.\n"
        )
        
        rec_tab = ctk.CTkTextbox(tabview.tab("Рекомендации"), wrap="word")
        rec_tab.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        rec_tab.insert("1.0", rec_text)
        rec_tab.configure(state="disabled")

if __name__ == "__main__":
    root = ctk.CTk()
    app = PhishingEDAApp(root)
    root.mainloop()