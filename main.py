import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd

import pathlib

names = ["NSGA-II/1219", "MO-MFEA/1019", "MO-MFEA-II/1019", "EMEA/1017", "Island_Model/1017"]
alg_names = [n.split("/")[0] for n in names]
original_names = ["Four bar truss design", "Reinforced concrete beam design", "Pressure vessel design","Hatch cover design","Coil compression spring design","Two bar truss design", "Welded beam design","Disc brake design","Vehicle crashworthiness design","Speed reducer design","Gear train design","Rocket injector design"]

def show_problem_info():
    info_array = []
    m = [2] * 5 + [3] * 7
    d = [4,3,4,2,3,3,4,4,5,7,4,4]
    for i, (name, m_, d_) in enumerate(zip(original_names, m, d)):
        info_array.append([int(i/2) + 1, name, m_, d_])
    df = pd.DataFrame(info_array, columns = ["Problem Set No.", "Problem Name", "Number of Objectives", "Number of Dimensions"], index = np.arange(len(info_array))+1)
    
    with st.expander("See Information of RE Problems"):
        st.dataframe(df)

@st.cache
def data_loader():

    pf = {}
    objs = {}
    metrics = {}

    problem_names = ["RE21", "RE22", "RE23", "RE24","RE25"]
    problem_names.extend(["RE31","RE32", "RE33", "RE34", "RE35", "RE36", "RE37"])

    for alg_name, n in zip(alg_names, names):

        objs[alg_name] = {}
        metrics[alg_name] = {}

        p = pathlib.Path(n)
        dirs = p.glob("*design/")

        for d in dirs:

            key = str(d.name).split("_")[-1]
            if key in original_names:
                objs[alg_name][key] = np.load(f'{n}/{d.name}/objectives.npz')["arr_0"]
                metrics[alg_name][key] = np.loadtxt(f'{n}/{d.name}/normalized_IGD.csv', delimiter = ",")

    for name, original_name in zip(problem_names, original_names):

        max_IGD = np.array([[np.max(alg_value[original_name])] for alg_value in metrics.values()]).max(axis = 0)

        for alg_value in metrics.values():
            alg_value[original_name] /= max_IGD

        front = np.loadtxt(f'real_world_problem/approximated_Pareto_fronts/{name}.csv', delimiter = ",")

        ideal = front.min(axis = 0)
        nadir = front.max(axis = 0)

        pf[original_name] = ((front[front[:, 0].argsort()] - ideal) / (nadir - ideal)).T

        for alg_name in alg_names:
            objs[alg_name][original_name] = (objs[alg_name][original_name] - ideal) / (nadir - ideal)

    return pf, objs, metrics


show_problem_info()
pf, objs, metrics = data_loader()

nobj = st.sidebar.radio("目的数", [2, 3])
if nobj == 2:
    problem = st.sidebar.selectbox("問題名", original_names[:5])
elif nobj == 3:
    problem = st.sidebar.selectbox("問題名", original_names[5:])
    st.sidebar.write("表示角度")
    z_angle = st.sidebar.slider("仰角", 0, 90, 30, 5)
    xy_angle = st.sidebar.slider("方位角( → 時計回り)", -360, 360, -60, 10)

alg = st.sidebar.radio("アルゴリズム名", alg_names)
gen = st.sidebar.slider("世代数", 1, 50)

class image_viewer:

    def __init__(self):

        self.color_palette = {}
        names = ["NSGA-II/1219", "MO-MFEA/1019", "MO-MFEA-II/1019", "EMEA/1017", "Island_Model/1017"]

        self.previous = None
        for alg_name, c in zip(names, ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']):

            n = alg_name.split("/")[0]
            self.color_palette[n] = c

    def view_image(self, nobj, pf, objs, metrics, gen, p_name = "Coil compression spring design", alg_name = "MO-MFEA"):

        if nobj == 2:
            fig, sub_ax = self.view_2d_image(pf, objs, metrics, gen, p_name, alg_name)
        elif nobj == 3:
            fig, sub_ax = self.view_3d_image(pf, objs, metrics, gen, p_name, alg_name)

        IGD_lines = sub_ax.plot(range(1, 51), np.array([m[p_name] for m in metrics.values()]).T, label = alg_names)
        current_gen = plt.axvline(gen, c = "k", label = "current gen", zorder = 4)

        for l, key in zip(IGD_lines, objs.keys()):

            if key == alg_name:
                l.zorder = 5
            else:
                l.zorder = 3
                l.set_linestyle("dashed")
                l.set_linewidth(1)

        sub_ax.legend()
        sub_ax.grid()
        plt.tight_layout()

        return fig

    def view_2d_image(self, pf, objs, metrics, gen, p_name = "Coil compression spring design", alg_name = "MO-MFEA"):

        fig, axes = plt.subplots(2, 1, sharex = False, sharey = False)
        ax, sub_ax = axes

        sub_ax.set_title("IGD (scaled [0-1])")

        # population plot
        ax.plot(pf[p_name][0], pf[p_name][1], c = "k", label = "Pareto Front")
        ax.set_aspect('equal')

        ax.set_title('population plot')
        ax.scatter(*objs[alg_name][p_name][gen - 1].T, label = "Population")

        ax.legend(loc = "upper right", bbox_to_anchor = [1.9, 1])
        ax.set(xlabel = r"$f_1$", ylabel = r"$f_2$")

        ax.set(xlim = [-0.05, 1.05], ylim = [-0.05, 1.05])

        return fig, sub_ax

    def view_3d_image(self, pf, objs, metrics, gen, p_name = "Welded beam design", alg_name = "MO-MFEA"):

        fig = plt.figure()
        ax = fig.add_subplot(211, projection = '3d')

        sub_ax = fig.add_subplot(212)
        sub_ax.set_title("\nIGD (scaled [0-1])")

        # population plot

        mask = ((objs[alg_name][p_name][gen - 1] < -0.05) | (objs[alg_name][p_name][gen - 1] > 1.05)).max(axis = 1)
        ax.scatter(*objs[alg_name][p_name][gen - 1][~mask].T, color = "tab:blue", alpha = 1, label = "Population")

        # pf plot
        ax.scatter(*pf[p_name], s = 1, color = "k", alpha = 1, label = "Pareto Front")

        ax.set_title('population plot')

        ax.legend(loc = "upper right", bbox_to_anchor = [2.0, 1])
        ax.set(xlabel = r"$f_1$", ylabel = r"$f_2$", zlabel = r"$f_3$")

        ax.set(xlim = [-0.05, 1.05], ylim = [-0.05, 1.05], zlim = [-0.05, 1.05])
        ax.view_init(elev = z_angle, azim = xy_angle)

        return fig, sub_ax


viewer = image_viewer()

fig = viewer.view_image(nobj, pf, objs, metrics, gen, problem, alg)

st.write(fig)

if nobj == 3:
    st.warning("matplotlibの仕様上，3目的の問題では個体とPFが重なって，うまく表示されない場合があります．")
