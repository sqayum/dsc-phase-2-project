import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from typing import Iterable, Optional, Union
from functools import reduce

class RegressionDataWrapper():
    def __init__(self, data: pd.DataFrame, *, target: str, discrete_vars: Optional[Iterable[str]] = None, categorical_vars: Optional[Iterable[str]] = None, row_index: Optional[str] = None, drop_cols: Optional[Iterable[str]] = None) -> None:

        self._df = data.drop_duplicates()

        if row_index is not None:
            if row_index not in self._df.columns:
                raise ValueError(f"column <{row_index}> not in dataset")
            self._df.set_index(row_index)

        if drop_cols is not None:
            self._df.drop(columns=drop_cols, inplace=True)

        self._vars = {"y": None,
                      "X":{"continuous": set(),
                           "discrete":set(),
                           "categorical": set()}}

        if target not in self._df.columns:
            raise ValueError(f"column <{target}> not in dataset")
        self._vars["y"] = target

        if discrete_vars is not None:
            for var in discrete_vars:
                self._vars["X"]["discrete"].add(var)

        if categorical_vars is not None:
            for var in categorical_vars:
                self._vars["X"]["categorical"].add(var)

        for var in set(self._df.columns) - (self._vars["X"]["discrete"] | self._vars["X"]["categorical"] | {self._vars["y"]}):
            self._vars["X"]["continuous"].add(var)

        self._split = False


    def _verify_predictors(self, names: Iterable[str]) -> Union[None, ValueError, AssertionError]:
        if not isinstance(names, pd.Index):
            names = pd.Index(names)
        if names.has_duplicates:
            raise AssertionError("duplicate columns not allowed")
        if self.target in names:
            raise AssertionError(f"column <{self.target}> is currently the target variable")
        if not set(names).issubset(self.predictors):
            raise ValueError(f"columns <{set(names).difference(self.predictors)}> not in dataset")

    def _check_validation(self):
        if self._split:
            raise AssertionError("cannot mutate dataset while validation is being performed")

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df):
        self._check_validation()
        self._df = df

    @property
    def target(self):
        return self._vars["y"]

    @property
    def predictors(self):
        return self._vars["X"]["continuous"] | self._vars["X"]["discrete"] | self._vars["X"]["categorical"]

    @property
    def cat_predictors(self):
        return self._vars["X"]["categorical"]

    @property
    def num_predictors(self):
        return self._vars["X"]["continuous"] | self._vars["X"]["discrete"]

    @property
    def cont_predictors(self):
        return self._vars["X"]["continuous"]

    @property
    def disc_predictors(self):
        return self._vars["X"]["discrete"]

    @property
    def y(self):
        return self._df[self.target]

    @property
    def X(self):
        return self._df[self.predictors]

    @property
    def X_cat(self):
        return self._df[self.cat_predictors]

    @property
    def X_num(self):
        return self._df[self.num_predictors]

    @property
    def X_cont(self):
        return self._df[self.cont_predictors]

    @property
    def X_disc(self):
        return self._df[self.disc_predictors]

    @property
    def df_standardized(self):
        return self._get_standardized()

    @property
    def shapes(self):
        return {"df": self._df.shape,
                "y": self.y.shape,
                "X": self.X.shape,
                "X_cat": self.X_cat.shape,
                "X_num": self.X_num.shape}


    def add_feature(self, feature: pd.Series, name: str, *, var_type: str) -> Union[None, AssertionError, ValueError]:
        self._check_validation()
        if feature.shape[0] != self._df.shape[0]:
            raise AssertionError(f"size of <{name}> are incompatible with dimensions of self._dfset")
        if name in self.predictors:
            raise AssertionError(f"<{name}> is already a predictor")
        if var_type in self._vars["X"].keys():
            self._vars["X"][var_type].add(name)
        else:
            raise ValueError(f"<{var_type}> is not a valid predictor type")
        feature.name = name
        self._df = pd.concat([self._df, feature], axis=1, join="inner")

    def remove_feature(self, name: str) -> Union[None, ValueError]:
        self._check_validation()
        if name not in self.predictors:
            raise ValueError(f"<{name}> is not a predictor")
        else:
            if name in self.cat_predictors:
                self._vars["X"]["categorical"].remove(name)
            elif name in self.disc_predictors:
                self._vars["X"]["discrete"].remove(name)
            else:
                self._vars["X"]["continuous"].remove(name)
            self._df.drop(columns=name, inplace=True)


    def _get_standardized(self) -> pd.DataFrame:
        scaler = StandardScaler()
        y = pd.DataFrame(self.y)
        X_standardized = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns, index=self.X.index)
        return y.join(X_standardized)


    def get_pairwise_corrs(self, variables=None, *, tol=None):
        if variables is not None:
            variables = pd.Index(variables)
            if self.target in variables:
                self._verify_predictors(variables.drop(self.target))
            else:
                self._verify_predictors(variables)
        else:
            variables = self._df.columns

        df_corr = self._df[variables].corr().abs().stack().reset_index().sort_values(0, ascending=False)
        df_corr["pairs"] = tuple(zip(df_corr.level_0, df_corr.level_1))
        df_corr.set_index("pairs", inplace=True)
        df_corr.drop(columns = ["level_0", "level_1"], inplace=True)
        df_corr.rename(columns={0: "correlation"}, inplace=True)
        df_corr.drop_duplicates(inplace=True)
        if tol is not None:
            return df_corr.loc[(df_corr.correlation >= tol) & (df_corr.correlation < 1)]
        else:
            return df_corr.loc[df_corr.correlation < 1]

    def get_VIF_dict(self, predictors=None):
        if predictors is not None:
            self._verify_predictors(predictors)
            X = self._df[pd.Index(predictors)]
        else:
            X = self.X
        X = sm.add_constant(X)
        vif_dict = {}
        for i, col in enumerate(X.columns):
            vif = variance_inflation_factor(X.values, i)
            vif_dict[col] = round(vif, 2)
        return {key: vif_dict[key] for key in sorted(vif_dict, key=lambda x: vif_dict[x], reverse=True)}

    def plot_corr_heatmap(self, variables=None, *, figsize=(20,30)):
        if variables is not None:
            if self.target in variables:
                self._verify_predictors(pd.Index(variables).drop(self.target))
            else:
                self._verify_predictors(variables)
        else:
            variables = self._df.columns

        df_corr = self._df[variables].corr().abs().round(3)
        mask = np.triu(np.ones_like(df_corr, dtype=bool))
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(df_corr, annot=True, mask=mask, cmap='Reds', ax=ax)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center",)
        plt.setp(ax.get_yticklabels(), rotation=0)
        ax.set_ylim(len(df_corr), 1)
        ax.set_xlim(xmax=len(df_corr)-1)
        fig.tight_layout()

    def corr_with_target(self, name):
        if name not in self._df.columns:
            raise ValueError(f"<{name}> is not a variable")
        else:
            return self._df.corr()[self.target][name]


    def plot_distribution(self, name:str):
        fig, ax = plt.subplots(figsize=(10,5))
        if name not in self._df.columns:
            raise ValueError(f"<{name}> is not a variable")
        if name in self.predictors:
            if name in self.cont_predictors:
                sns.histplot(data=self._df, x=name, ax=ax, kde=True)
            else:
                sns.countplot(data=self._df, x=name, ax=ax, color="tab:blue")
        else:
            sns.histplot(data=self._df, x=name, ax=ax, kde=True)
        fig.tight_layout()

    def plot_predictor(self, name: str) -> Union[None, ValueError]:
        if name not in self._df.columns:
            raise ValueError(f"<{name}> is not a variable")
        x = self._df[name]
        y = self._df[self.target]
        if name in self.cont_predictors:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15,8), sharex=True)
            sns.histplot(x=x, ax=ax1, kde=True, color="tab:blue")
            sns.regplot(x=x, y=y, ax=ax2, ci=None, color="tab:blue", line_kws={"color": "salmon", "linestyle": "--", "linewidth": 1.5})

        else:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15,12))
            sns.countplot(x=x, ax=ax1, color="tab:blue")
            sns.boxplot(x=x, y=y, ax=ax2, showfliers=False, width=0.5, color="tab:blue")
            # ax3.set_xlim(ax2.xlim[0], ax2.xlim[1])
            if name in self.disc_predictors:
                sns.regplot(x=x, y=y, ax=ax3, ci=None, truncate=False, color="tab:blue", line_kws={"color": "salmon", "linestyle": "--", "linewidth": 1.5})
            else:
                sns.pointplot(x=x, y=y, ax=ax3, x_estimator=np.median, color="salmon")
                # ax3.set_ylim(ax2.ylim[0], ax2.ylim[1])
            ax1.set_xlabel("")
            ax2.set_xlabel("")
            plt.tight_layout()


    def log_transform(self, name):
        if name not in self._df.columns:
            raise ValueError(f"<{name}> is not a variable")
        if "log_"+name in self._df.columns:
            return
        else:
            if name in self.predictors:
                if name in self.cont_predictors:
                    self._vars["X"]["continuous"].remove(name)
                    self._vars["X"]["continuous"].add("log_"+name)
                elif name in self.disc_predictors:
                    self._vars["X"]["discrete"].remove(name)
                    self._vars["X"]["discrete"].add("log_"+name)
                else:
                    raise TypeError(f"cannot log transform categorical variable <{name}>")
            else:
                self._vars["y"] = "log_" + self._vars["y"]
            self._df[name] = np.log(self._df[name])
            self._df.rename(columns={name: "log_"+name}, inplace=True)

    def inv_log_transform(self, name):
        var_name = name.split('log_')[-1]
        if name not in self._df.columns:
            raise ValueError(f"<{name}> is not a variable")
        if var_name in self._df.columns:
            return
        else:
            if name in self.predictors:
                if name in self.cont_predictors:
                    self._vars["X"]["continuous"].remove(name)
                    self._vars["X"]["continuous"].add("log"+name)
                elif name in self.disc_predictors:
                    self._vars["X"]["discrete"].remove(name)
                    self._vars["X"]["discrete"].add("log"+name)
                else:
                    raise TypeError(f"<{name}> is not numeric")
            else:
                self._vars["y"] = var_name
            self._df[name] = np.exp(self._df[name])
            self._df.rename(columns={name: var_name}, inplace=True)


    def remove_outliers(self, name):
        self._check_validation()
        if name not in self._df.columns:
            raise ValueError(f"<{name}> is not a variable")
        self._df = self._df[(np.abs(stats.zscore(self._df[name])) < 3)]


    def one_hot_encode(self, name: str):
        self._check_validation()
        if name not in self.cat_predictors:
            return ValueError(f"<{name}> is not a categorical variable")
        ohe = OneHotEncoder(drop="first", sparse=False)
        var_ohe = ohe.fit_transform(self._df[[name]])
        var_cols = ohe.get_feature_names(input_features=[name])
        var_df = pd.DataFrame(var_ohe, columns=var_cols, index=self._df.index).astype("uint64")

        self._df = self._df.join(var_df, how="inner")
        self._df = self._df.reset_index().drop_duplicates("id").set_index("id")

        self._vars["X"]["categorical"] = self._vars["X"]["categorical"] | set(var_cols) | {name}

    def get_formula(self, predictors: Iterable[str]) -> str:
        self._verify_predictors(predictors)
        return self.target + " ~ " + ' + '.join(predictor for predictor in predictors)

    def start_traintest(self):
        if self._split:
            return
        self._split = True

    def validate(self, params):
        if not self._split:
            raise AssertionError("dataset is not train-test split")
        self._verify_predictors(params)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        scaler.fit_transform(X_train)
        scaler.transform(X_test)

        linreg = LinearRegression()
        linreg.fit(X_train, y_train)
        y_train_pred = linreg.predict(X_train)
        y_test_pred = linreg.predict(X_test)

        metrics = {}

        mse_train = mean_squared_error(y_train.copy(), y_train_pred)
        mse_test = mean_squared_error(y_test.copy(), y_test_pred)
        mse_error = np.abs(mse_test - mse_train)
        metrics["MSE"] = {"train": mse_train, "test": mse_test, "|difference|": mse_error}

        mae_train = mean_absolute_error(y_train.copy(), y_train_pred)
        mae_test = mean_absolute_error(y_test.copy(), y_test_pred)
        mae_error = np.abs(mae_test - mae_train)
        metrics["MAE"] = {"train": mae_train, "test": mae_test, "|difference|": mae_error}

        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)
        metrics["RMSE"] = {"train": rmse_train, "test": rmse_test, "|difference|": np.abs(rmse_test - rmse_train)}

        return pd.DataFrame(metrics)

    def end_traintest(self):
        if not self._split:
            return
        self._split = False

def ols_regression(data, predictors: Iterable[str]):
    f = data.get_formula(predictors)
    return ols(formula=f, data=data.df_standardized).fit()



def plot_residuals(model):
    residuals = model.resid

    fig1, ax = plt.subplots(figsize=(12,5))
    x = np.linspace(0, 1, len(model.resid))
    sns.scatterplot(x=x, y=stats.zscore(residuals), ax=ax, alpha=0.4)
    sns.lineplot(x=x, y=np.zeros_like(x), ax=ax, color="black")
    ax.set_xlabel("Fitted")
    ax.set_ylabel("Standardized Residuals")
    ax.set_title("Scatterplot of Standardized Residuals")



    fig2, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,9), sharex=True)
    ax1.set_xlabel("Standardized Residuals")
    sns.histplot(x=stats.zscore(residuals), kde=True, ax=ax1)
    ax1.set_title("Histogram of Standardized Residuals")
    sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True, ax=ax2)
    ax2.set_title("Normal Q-Q Plot of Standardized Residuals")
    plt.show()
