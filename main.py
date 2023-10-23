import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

yes, no = 0, 1  # Index values

class GenericAnalysis:
    def __init__(self, x_probs, x_states, y_states, y_prob):
        self.x_probs = x_probs
        self.x_states = x_states
        self.t_st = y_states
        self.t_prob = y_prob

        self.mean_X = None
        self.var_X = None
        self.std_X = None

        self.Y_means = None
        self.Y_vars = None
        self.Y_stds = None

        self.Y_mean = None
        self.Y_var = None
        self.Y_std = None

    def compute_statistics(self):
        self.mean_X = np.sum(self.x_probs * self.x_states)
        self.var_X = np.sum(np.square(self.x_states - self.mean_X) * self.x_probs)
        self.std_X = np.sqrt(self.var_X)

        self.Y_means = self.x_probs[yes] * self.t_st
        self.Y_vars = self.x_probs[yes] * (1 - self.x_probs[yes]) * self.t_st
        self.Y_stds = np.sqrt(self.Y_vars)

        # Y's mean, variance, and standard deviation
        self.Y_mean = np.sum(self.Y_means * self.t_prob)
        self.Y_var = (
            np.sum(self.t_prob * (self.Y_means**2 + self.Y_stds**2))
            - self.Y_mean**2
        )
        self.Y_std = np.sqrt(self.Y_var)

class TravelAnalysis(GenericAnalysis):
    def compute_statistics(self):
        super().compute_statistics()
        self.transports = {
            "Car": {"multiplier": 1},
            "Cycling": {"multiplier": 0.78},
            "Subway": {"multiplier": 2},
            "Bus": {"multiplier": 2},
            "Walking": {"multiplier": 2.9},
        }
        self.U = {}
        for x,y in self.transports.items():
            self.U[x] = y["multiplier"] * self.Y_mean


class CO2Analysis(GenericAnalysis):
    def compute_statistics(self):
        super().compute_statistics()
        self.U = self.Y_mean * 171

class DisabledUserAnalysis(GenericAnalysis):
    def compute_statistics(self):
        super().compute_statistics()
        self.UFunc = {
            "NonDisabled": {"multiplier": 1},
            "Disabled": {"multiplier": 0.7},
        }
        self.U = {}

        for x,y in self.UFunc.items():
            self.U[x] = y["multiplier"] * self.Y_mean

class CollisionAnalysis(GenericAnalysis):
    def compute_statistics(self):
        super().compute_statistics()
        self.UFunc = {
            # "NonDisabled": {"multiplier": 1},
            # "Disabled": {"multiplier": 0.7},
        }
        self.U = {}

        for x,y in self.UFunc.items():
            self.U[x] = y["multiplier"] * self.Y_mean


# =============================================================================
# =================================  CO2 ======================================
# =============================================================================
analysis_CO2_initial = CO2Analysis(
    x_probs=np.array([0.674, 0.326]),
    x_states=np.array([1, 0]),
    y_states=np.array([900, 1122, 1350]),
    y_prob=np.array([0.1, 0.2, 0.7]),
)
analysis_CO2_ltn1 = CO2Analysis(
    x_probs=np.array([0.4, 0.6]),
    x_states=np.array([1, 0]),
    y_states=np.array([900, 1122, 1350]),
    y_prob=np.array([0.2, 0.3, 0.5]),
)
analysis_CO2_ltn2 = CO2Analysis(
    x_probs=np.array([0.65, 0.35]),
    x_states=np.array([1, 0]),
    y_states=np.array([900, 1122, 1350]),
    y_prob=np.array([0.05, 0.15, 0.8]),
)
analysis_CO2_initial.compute_statistics()
analysis_CO2_ltn1.compute_statistics()
analysis_CO2_ltn2.compute_statistics()
print('C02 Analysis:')
print(f"Current layout:{analysis_CO2_initial.U}")
print(f"LTN Plan 1: {analysis_CO2_ltn1.U}")
print(f"LTN Plan 2: {analysis_CO2_ltn2.U}")
print('\n')
# =============================================================================
# ===========================  Duration of travel =============================
# =============================================================================
analysis_travel_initial = TravelAnalysis(
    x_probs=np.array([0.674, 0.326]),
    x_states=np.array([1, 0]),
    y_states=np.array([900, 1122, 1350]),
    y_prob=np.array([0.1, 0.2, 0.7]),
)
analysis_travel_ltn1 = TravelAnalysis(
    x_probs=np.array([0.4, 0.6]),
    x_states=np.array([1, 0]),
    y_states=np.array([900, 1122, 1350]),
    y_prob=np.array([0.2, 0.3, 0.5]),
)
analysis_travel_ltn2 = TravelAnalysis(
    x_probs=np.array([0.65, 0.35]),
    x_states=np.array([1, 0]),
    y_states=np.array([900, 1122, 1350]),
    y_prob=np.array([0.05, 0.15, 0.8]),
)
analysis_travel_initial.compute_statistics()
analysis_travel_ltn1.compute_statistics()
analysis_travel_ltn2.compute_statistics()
print('Travel Analysis:')
print(f"Current layout:{analysis_travel_initial.U}")
print(f"LTN Plan 1: {analysis_travel_ltn1.U}")
print(f"LTN Plan 2: {analysis_travel_ltn2.U}")
print('\n')
# =============================================================================
# ============================  Disabled Users =================================
# =============================================================================
analysis_disabled_initial = DisabledUserAnalysis(
    x_probs=np.array([0.62, 0.38]),
    x_states=np.array([1, 0]),
    y_states=np.array([900, 1122, 1350]),
    y_prob=np.array([0.1, 0.2, 0.7]),
)
analysis_disabled_ltn1 = DisabledUserAnalysis(
    x_probs=np.array([0.7, 0.3]),
    x_states=np.array([1, 0]),
    y_states=np.array([900, 1122, 1350]),
    y_prob=np.array([0.2, 0.3, 0.5]),
)
analysis_disabled_ltn2 = DisabledUserAnalysis(
    x_probs=np.array([0.55, 0.45]),
    x_states=np.array([1, 0]),
    y_states=np.array([900, 1122, 1350]),
    y_prob=np.array([0.05, 0.15, 0.8]),
)
analysis_disabled_initial.compute_statistics()
analysis_disabled_ltn1.compute_statistics()
analysis_disabled_ltn2.compute_statistics()
print('Disabled Users Analysis:')
print(f"Current layout:{analysis_disabled_initial.U}")
print(f"LTN Plan 1: {analysis_disabled_ltn1.U}")
print(f"LTN Plan 2: {analysis_disabled_ltn2.U}")
print('\n')
# =============================================================================
# ===============================  Collisions =================================
# =============================================================================
analysis_collisions_initial = CollisionAnalysis(
    x_probs=np.array([0.744, 0.256]),
    x_states=np.array([1, 0]),
    y_states=np.array([900, 1122, 1350]),
    y_prob=np.array([0.1, 0.2, 0.7]),
)
analysis_collisions_ltn1 = CollisionAnalysis(
    x_probs=np.array([0.2, 0.8]),
    x_states=np.array([1, 0]),
    y_states=np.array([900, 1122, 1350]),
    y_prob=np.array([0.2, 0.3, 0.5]),
)
analysis_collisions_ltn2 = CollisionAnalysis(
    x_probs=np.array([0.8, 0.2]),
    x_states=np.array([1, 0]),
    y_states=np.array([900, 1122, 1350]),
    y_prob=np.array([0.05, 0.15, 0.8]),
)
analysis_collisions_initial.compute_statistics()
analysis_collisions_ltn1.compute_statistics()
analysis_collisions_ltn2.compute_statistics()
print('Collision Analysis:')
print(f"Current layout:{analysis_collisions_initial.U}")
print(f"LTN Plan 1: {analysis_collisions_ltn1.U}")
print(f"LTN Plan 2: {analysis_collisions_ltn2.U}")
print('\n')

val = 690
intervals_bnd = np.linspace(0, 1200, 200 + 1)  # For 100 intervals

class CombinedHistogramPlotter:
    def __init__(self, analyses, xlabel, ylabel, title, intervals_bnd, colors, labels, filename: str):

        self.analyses = analyses

        self.xlabel = xlabel

        self.ylabel = ylabel

        self.title = title

        self.intervals_bnd = intervals_bnd

        self.colors = colors

        self.labels = labels

        self.filename = filename


        self.probs = [self._calculate_probs(analysis) for analysis in analyses]
        self.intervals_mean = 0.5 * (self.intervals_bnd[:-1] + self.intervals_bnd[1:])
        self.val_50_percent = [self._calculate_50_percent_value(prob) for prob in self.probs]


    def _calculate_probs(self, analysis):
        no_itv = len(self.intervals_bnd) - 1
        probs = np.zeros(no_itv)

        for ii in range(no_itv):
            val_i_up = self.intervals_bnd[ii + 1]
            val_i_down = self.intervals_bnd[ii]

            val_probs_i_up = norm.cdf((val_i_up - analysis.Y_means) / analysis.Y_stds)
            val_probs_i_down = norm.cdf((val_i_down - analysis.Y_means) / analysis.Y_stds)

            val_prob1_i = np.sum(analysis.t_prob * (val_probs_i_up - val_probs_i_down))
            probs[ii] = val_prob1_i

        return probs

    def _calculate_50_percent_value(self, prob):
        cumulative_probs = np.cumsum(prob)
        index_50_percent = np.argmax(cumulative_probs >= 0.5)
        return self.intervals_mean[index_50_percent]


    def plot(self):
        plt.figure(figsize=(10, 6))
        width = (self.intervals_bnd[-1] - self.intervals_bnd[0]) / len(self.intervals_mean)

        bottom = np.zeros_like(self.intervals_mean)
        for i, (prob, color, label, val_50_percent) in enumerate(zip(self.probs, self.colors, self.labels, self.val_50_percent)):
            plt.bar(self.intervals_mean, prob, width=width, bottom=bottom, color=color, label=label)
            bottom += prob
            plt.axvline(x=val_50_percent, color=color, linestyle='--', label=f"50% - {label}")


        plt.grid(True)
        plt.margins(0)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.legend(loc="upper left")
        plt.tight_layout()
        if self.filename:
            downloads_folder = os.path.join(os.path.expanduser('~'), 'Downloads')
            file_path = os.path.join(downloads_folder, self.filename)
            plt.savefig(file_path, format="pdf")

        else:
            plt.show()


# Create an instance of the CombinedHistogramPlotter class for the three scenarios
plotter_combined = CombinedHistogramPlotter(
    analyses=[analysis_travel_initial, analysis_travel_ltn1, analysis_travel_ltn2],
    xlabel="Travels / hour",
    ylabel="Probability",
    title="Combined Scenario - Travels / hour",
    intervals_bnd=intervals_bnd,
    colors=["limegreen", "skyblue", "lightcoral"],
    labels=["Original", "LTN1", "LTN2"],
    filename = r"travel_and_CO2.pdf"
)

plotter_combined.plot()


plotter_2_combined = CombinedHistogramPlotter(
    analyses=[analysis_disabled_initial, analysis_disabled_ltn1, analysis_disabled_ltn2],
    xlabel="Disabled Users / hour",
    ylabel="Probability",
    title="Combined Scenario - Disabled Users / hour",
    intervals_bnd=intervals_bnd,
    colors=["limegreen", "skyblue", "lightcoral"],
    labels=["Original", "LTN1", "LTN2"],
    filename = r"disabled_users.pdf"
)
plotter_2_combined.plot()



plotter_3_combined = CombinedHistogramPlotter(
    analyses=[analysis_collisions_initial, analysis_collisions_ltn1, analysis_collisions_ltn2],
    xlabel="Collisions / hour",
    ylabel="Probability",
    title="Combined Scenario - Collisions / hour",
    intervals_bnd=intervals_bnd,
    colors=["limegreen", "skyblue", "lightcoral"],
    labels=["Original", "LTN1", "LTN2"],
    filename = r"collisions.pdf"
)
plotter_3_combined.plot()