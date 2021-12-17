import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd

import pathlib

alg_names = [n.split("/")[0] for n in["NSGA-II/1017", "MO-MFEA/1019", "MO-MFEA-II/1019", "EMEA/1017", "Island_Model/1017"]]
original_names = ["Four bar truss design", "Reinforced concrete beam design", "Pressure vessel design","Hatch cover design","Coil compression spring design"]

def show_problem_info():
    info_array = []
    d = [4,3,4,2,3]
    for i, (name, d_) in enumerate(zip(original_names, d)):
        info_array.append([int(i/2) + 1, name, 2, d_])
    df = pd.DataFrame(info_array, columns = ["Problem Set No.", "Problem Name", "Number of Objectives", "Number of Dimensions"], index = np.arange(len(info_array))+1)
    
    st.dataframe(df)

@st.cache
def data_loader():


    pf = {}
    objs = {}
    metrics = {}

    problem_names = ["RE21", "RE22", "RE23", "RE24","RE25"]

    names = ["NSGA-II/1017", "MO-MFEA/1019", "MO-MFEA-II/1019", "EMEA/1017", "Island_Model/1017"]

    for alg_name in names:

        n = alg_name.split("/")[0]
        objs[n] = {}
        metrics[n] = {}

        p = pathlib.Path(alg_name)
        dirs = p.glob("*design/")

        for d in dirs:

            key = str(d.name).split("_")[-1]
            if key in original_names:
                objs[n][key] = np.load(f'{alg_name}/{d.name}/objectives.npz')["arr_0"]
                metrics[n][key] = np.loadtxt(f'{alg_name}/{d.name}/normalized_IGD.csv', delimiter = ",")

    # original_names = ["Four bar truss design", "Reinforced concrete beam design", "Pressure vessel design","Hatch cover design","Coil compression spring design","Two bar truss design", "Welded beam design","Disc brake design","Vehicle crashworthiness design","Speed reducer design","Gear train design","Rocket injector design","Car side impact design","Conceptual marine design"]

    for name, original_name in zip(problem_names, original_names):

        max_IGD = np.array([[np.max(alg_value[original_name])] for alg_value in metrics.values()]).max(axis = 0)

        for alg_value in metrics.values():
            alg_value[original_name] /= max_IGD

        front = np.loadtxt(f'real_world_problem/approximated_Pareto_fronts/{name}.csv', delimiter = ",")

        ideal = front.min(axis = 0)
        nadir = front.max(axis = 0)

        pf[original_name] = ((front[front[:, 0].argsort()] - ideal) / (nadir - ideal)).T

        for alg_name in alg_names:
            objs[alg_name.split("/")[0]][original_name] = (objs[alg_name.split("/")[0]][original_name] - ideal) / (nadir - ideal)

    return pf, objs, metrics

# @st.cache
def first_draw(pf, metrics):

    p_name = "Coil compression spring design"
    alg_name = "MO-MFEA"

    # setting
    figs = dict(zip(["fig", "axes"], plt.subplots(2, 1)))
    ax, sub_ax = figs["axes"]
    figs["ax"] = ax
    figs["sub_ax"] = sub_ax

    # population plot
    figs["line"], = figs["ax"].plot(pf[p_name][0], pf[p_name][1], c = "k", label = "Pareto Front")
    figs["ax"].set_aspect('equal')

    figs["ax"].set_title('population plot')
    figs["scat"] = figs["ax"].scatter([], [], label = "Population")

    figs["ax"].legend(loc = "upper right", bbox_to_anchor = [1.9, 1])

    # IGD plot
    figs["IGD_lines"] = sub_ax.plot(range(1, 51), np.array([m[p_name] for m in metrics.values()]).T, label = alg_names)
    figs["current_gen"] = plt.axvline(0, c = "k", label = "current gen")
    figs["sub_ax"].set_title("IGD (scaled [0-1])")

    figs["sub_ax"].legend()
    figs["sub_ax"].grid()

    plt.tight_layout()

    return figs

class image_viewer:

    def __init__(self):

        self.color_palette = {}
        names = ["NSGA-II/1017", "MO-MFEA/1019", "MO-MFEA-II/1019", "EMEA/1017", "Island_Model/1017"]

        self.previous = None
        for alg_name, c in zip(names, ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']):

            n = alg_name.split("/")[0]
            self.color_palette[n] = c

    def view_image(self, figs, pf, objs, metrics, gen, p_name = "Coil compression spring design", alg_name = "MO-MFEA"):

        figs["line"].set_data(pf[p_name][0], pf[p_name][1])
        figs["scat"].set_offsets(objs[alg_name][p_name][gen - 1])

        for l, key in zip(figs["IGD_lines"], objs.keys()):

            if key == alg_name:
                l.set_data(range(1, 51), metrics[key][p_name])
                l.zorder = 5
            else:
                l.set_data(range(1, 51), metrics[key][p_name])
                l.zorder = -1
                l.set_linestyle("dashed")
                l.set_linewidth(1)

#         self.IGD_lines.set_data(range(1, 51), self.metrics[alg_name][p_name])
#         self.IGD_lines.set_color(self.color_palette[alg_name])
#         self.current_point.sef_offsets(gen, self.metrics[alg_name][p_name][gen - 1])
        figs["current_gen"].set_xdata(gen)

show_problem_info()

pf, objs, metrics = data_loader()

viewer = image_viewer()

problem = st.sidebar.selectbox("問題名", original_names)
alg = st.sidebar.radio("アルゴリズム名", alg_names)
gen = st.sidebar.slider("世代数", 1, 50)

fig_dict = first_draw(pf, metrics)
viewer.view_image(fig_dict, pf, objs, metrics, gen, problem, alg)

st.write(fig_dict["fig"])

# viewer.first_draw(problem.value, alg1.value)
# widgets.interact(viewer.view_image, gen = (1, 50, 1), p_name = problem, alg_name = alg1)


# alg_names = [n.split("/")[0] for n in["NSGA-II/1017", "MO-MFEA/1019", "MO-MFEA-II/1019", "EMEA/1017", "Island_Model/1017"]]
# original_names = ["Two bar truss design", "Welded beam design","Disc brake design","Vehicle crashworthiness design","Speed reducer design","Gear train design","Rocket injector design","Car side impact design","Conceptual marine design"]

# class image_viewer:

#     def __init__(self):

#         self.pf = {}
#         self.objs = {}
#         self.metrics = {}

#         self.previous = None

#         problem_names = ["RE31","RE32", "RE33", "RE34", "RE35", "RE36", "RE37"]

#         alg_names = ["NSGA-II/1017", "MO-MFEA/1019", "MO-MFEA-II/1019", "EMEA/1017", "Island_Model/1017"]

#         for alg_name in alg_names:

#             n = alg_name.split("/")[0]
#             self.objs[n] = {}
#             self.metrics[n] = {}

#             p = pathlib.Path(alg_name)
#             dirs = p.glob("*design/")

#             for d in dirs:

#                 key = str(d.name).split("_")[-1]
#                 self.objs[n][key] = np.load(f'{alg_name}/{d.name}/trial1_objectives.npz')["arr_0"]
#                 self.metrics[n][key] = np.loadtxt(f'{alg_name}/{d.name}/normalized_IGD_log_trial1.csv', delimiter = ",")

#         original_names = ["Two bar truss design", "Welded beam design","Disc brake design","Vehicle crashworthiness design","Speed reducer design","Gear train design","Rocket injector design","Car side impact design","Conceptual marine design"]

#         for name, original_name in zip(problem_names, original_names):

#             pf = np.loadtxt(f'real_world_problem/approximated_Pareto_fronts/{name}.csv', delimiter = ",")

#             ideal = pf.min(axis = 0)
#             nadir = pf.max(axis = 0)

#             self.pf[original_name] = ((pf[pf[:, 0].argsort()] - ideal) / (nadir - ideal)).T

#             for alg_name in alg_names:
#                 self.objs[alg_name.split("/")[0]][original_name] = (self.objs[alg_name.split("/")[0]][original_name] - ideal) / (nadir - ideal)

#     def first_draw(self, p_name = "Welded beam design", alg_name = "MO-MFEA"):

#         self.fig = plt.figure()
#         self.ax = self.fig.add_subplot(111, projection = '3d')
#         self.line = self.ax.scatter(*self.pf[p_name], color = "k", alpha = 1, label = "Pareto Front")

# #         self.ax.set_aspect('equal')

#         self.txt = self.fig.suptitle(f'IGD: {self.metrics[alg_name][p_name][0]:.4e}')
#         self.scat = self.ax.scatter([], [], [], label = "Population")

#         self.fig.legend()

#     def view_image(self, gen, p_name = "Welded beam design", alg_name = "MO-MFEA"):

# #         if self.previous != p_name:
# #             self.first_draw(p_name, alg_name)
# #             self.previous = p_name

# #         self.first_draw(p_name, alg_name)

#         self.line.remove()
#         self.scat.remove()

#         mask = ((self.objs[alg_name][p_name][gen - 1] < -0.05) | (self.objs[alg_name][p_name][gen - 1] > 1.05)).max(axis = 1)

#         self.txt.set_text(f'{alg_name.replace("_", " ")}\nIGD: {self.metrics[alg_name][p_name][gen - 1]:.4e}')
#         self.line = self.ax.scatter(*self.pf[p_name], color = "k", alpha = 1, s = 1, marker = "s")
#         self.scat = self.ax.scatter(*self.objs[alg_name][p_name][gen - 1][~mask].T, color = "tab:blue", alpha = 1)

#         self.ax.set(xlim = [-0.05, 1.05], ylim = [-0.05, 1.05], zlim = [-0.05, 1.05])

# viewer = image_viewer()

# problem = widgets.Dropdown(description = "問題名", options = original_names)
# alg1 = widgets.ToggleButtons(description = "アルゴリズム名", options = alg_names)

# viewer.first_draw(problem.value, alg1.value)
# widgets.interact(viewer.view_image, gen = (1, 50, 1), p_name = problem, alg_name = alg1);
