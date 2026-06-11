import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json
import math

def run_pareto_mas_optimization():
    # ---------------- 1. 基础参数与网络拓扑 ----------------
    roles = ["A", "B", "C", "D", "E"] 
    workflows = ["W1", "W2"]
    regions = [f"R{i}" for i in range(1, 9)]
    
    mas_edges = [("A", "B"), ("B", "C"), ("B", "D"), ("C", "E"), ("D", "E")]
    instances = {role: [f"{role}_{idx}" for idx in range(1, 4)] for role in roles}
    all_instances = [i for role_insts in instances.values() for i in role_insts]
    
    physical_edges = [
        ("R1", "R2"), ("R2", "R3"), ("R3", "R4"), ("R4", "R1"),
        ("R5", "R6"), ("R6", "R7"), ("R7", "R8"), ("R8", "R5"),
        ("R1", "R5"), ("R2", "R6"), ("R3", "R7"), ("R4", "R8")
    ]
    
    # 带宽费用矩阵/字典 (各不相同，范围1-25)
    edge_costs = {
        ("R1", "R2"): 12.0, ("R2", "R3"): 18.0, ("R3", "R4"): 22.0, ("R4", "R1"): 10.0,
        ("R5", "R6"): 15.0, ("R6", "R7"): 8.0,  ("R7", "R8"): 14.0, ("R8", "R5"): 20.0,
        ("R1", "R5"): 5.0,  ("R2", "R6"): 25.0, ("R3", "R7"): 16.0, ("R4", "R8"): 7.0 
    }
    
    G_topo = nx.Graph()
    G_topo.add_nodes_from(regions)
    for (u, v) in physical_edges:
        cost = edge_costs.get((u, v), edge_costs.get((v, u), 20.0))
        G_topo.add_edge(u, v, weight=cost)
        
    # ---------------- 2. CVaR 故障情景集合与概率定义 (精简版) ----------------
    scenarios = ["s0", "s1"]
    
    # 调整概率保证总和为 1
    prob = {
        "s0": 0.90, # 正常运行，无故障
        "s1": 0.10  # 仅测试一个典型节点故障：核心节点 R5 宕机
    }
    
    # 节点可用性矩阵 A_rs
    A_rs = {r: {s: 1 for s in scenarios} for r in regions}
    A_rs["R5"]["s1"] = 0  # 仅在 s1 情景下 R5 发生故障

    # 链路故障拓扑字典（清空以消除干扰）
    link_faults = {}

    shortest_bw_costs = {s: {} for s in scenarios}
    shortest_hops = {s: {} for s in scenarios}

    # 各故障场景下图底座拓扑的自适应预处理
    for s in scenarios:
        G_s = G_topo.copy()
        if s in link_faults:
            u, v = link_faults[s]
            if G_s.has_edge(u, v): G_s.remove_edge(u, v)
        for r in regions:
            if A_rs[r][s] == 0:
                G_s.remove_edges_from(list(G_s.edges(r)))
                
        for r1 in regions:
            shortest_bw_costs[s][r1] = {}
            shortest_hops[s][r1] = {}
            for r2 in regions:
                if A_rs[r1][s] == 0 or A_rs[r2][s] == 0:
                    shortest_bw_costs[s][r1][r2] = 10000.0 
                    shortest_hops[s][r1][r2] = 10000.0
                else:
                    try:
                        shortest_bw_costs[s][r1][r2] = nx.dijkstra_path_length(G_s, r1, r2, weight='weight')
                        shortest_hops[s][r1][r2] = nx.shortest_path_length(G_s, r1, r2)
                    except nx.NetworkXNoPath:
                        shortest_bw_costs[s][r1][r2] = 10000.0
                        shortest_hops[s][r1][r2] = 10000.0

    def get_bw_cost(s, r1, r2): return shortest_bw_costs[s][r1][r2]
    def get_trans_delay(s, r1, r2): return 1.5 * shortest_hops[s][r1][r2]
    
    elec_cost = {"R1":25, "R2":20, "R3":18, "R4":15, "R5":12, "R6":10, "R7":8, "R8":2}
    base_time = {"A": 3.0, "B": 4.0, "C": 5.0, "D": 5.0, "E": 6.0}
    M = 1000.0 

    kv_savings = {
        ("A", "B"): 2.5,
        ("B", "C"): 3.0, ("A", "C"): 1.5, ("D", "C"): 2.0,
        ("B", "D"): 3.0, ("A", "D"): 1.5, ("C", "D"): 2.0,
        ("D", "E"): 2.5, ("C", "E"): 2.5, ("B", "E"): 1.5, ("A", "E"): 1.0
    }

    # ---------------- 3. 全局构建 Gurobi 数学规划模型 ----------------
    model = gp.Model("MAS_CVaR_Optimization")
    model.setParam('OutputFlag', 0)
    
    # 第一阶段决策变量 (硬核物理部署，跨情景不可变)
    d = model.addVars(all_instances, regions, vtype=GRB.BINARY, name="d")
    
    # 第二阶段决策变量 (自适应弹性路由与调度，全面注入 s 场景空间)
    x = model.addVars(workflows, all_instances, scenarios, vtype=GRB.BINARY, name="x")
    z = model.addVars(workflows, roles, regions, scenarios, vtype=GRB.BINARY, name="z")
    
    S = model.addVars(workflows, roles, scenarios, vtype=GRB.CONTINUOUS, name="S")
    C = model.addVars(workflows, roles, scenarios, vtype=GRB.CONTINUOUS, name="C")
    D_var = model.addVars(workflows, roles, scenarios, vtype=GRB.CONTINUOUS, name="D")
    
    # CVaR (条件风险价值) 相关决策元
    alpha_cvar = 0.95
    eta = model.addVar(vtype=GRB.CONTINUOUS, name="eta") # VaR 阈值分位数
    u_vars = model.addVars(scenarios, vtype=GRB.CONTINUOUS, lb=0.0, name="u_vars") # 尾部损失超出量
    T_max_s = model.addVars(scenarios, vtype=GRB.CONTINUOUS, name="T_max_s") # 每种场景下的最长耗时
    
    CVaR_T = model.addVar(vtype=GRB.CONTINUOUS, name="CVaR_T") # 目标1：Makespan 的 CVaR
    Expected_Cost = model.addVar(vtype=GRB.CONTINUOUS, name="Expected_Cost") # 目标2：期望运营成本
    
    b = model.addVars(workflows, roles, workflows, roles, scenarios, vtype=GRB.BINARY, name="b")
    o = model.addVars(workflows, roles, workflows, roles, scenarios, vtype=GRB.BINARY, name="o")
    is_ov1 = model.addVars(workflows, roles, scenarios, vtype=GRB.BINARY, name="is_ov1")
    is_ov2 = model.addVars(workflows, roles, scenarios, vtype=GRB.BINARY, name="is_ov2")
    share_kv = model.addVars(workflows, roles, roles, scenarios, vtype=GRB.BINARY, name="share_kv")
    active_save = model.addVars(workflows, roles, roles, scenarios, vtype=GRB.BINARY, name="active_save")

    # 第一阶段基本物理硬约束 
    model.addConstrs((gp.quicksum(d[i, r] for r in regions) == 1 for i in all_instances))
    model.addConstrs((gp.quicksum(d[i, r] for i in all_instances) <= 2 for r in regions))
    
    # 注入情景层面的核心约束循环
    for s in scenarios:
        for w in workflows:
            for role in roles:
                model.addConstr(gp.quicksum(x[w, i, s] for i in instances[role]) == 1)
        model.addConstrs((gp.quicksum(x[w, i, s] for w in workflows) <= 1 for i in all_instances))

        for w in workflows:
            for role in roles:
                for r in regions:
                    model.addConstr(z[w, role, r, s] == gp.quicksum(d[i, r] * x[w, i, s] for i in instances[role]))
                    # 【故障隔离屏障】：如果该区域宕机 (A_rs=0)，则强制限制 z=0，强迫系统寻找备份实例
                    model.addConstr(z[w, role, r, s] <= A_rs[r][s])

        # 时序控制与并发碰撞判定 (场景隔离)
        for w1 in workflows:
            for a1 in roles:
                for w2 in workflows:
                    for a2 in roles:
                        if w1 == w2 and a1 == a2:
                            model.addConstr(o[w1, a1, w2, a2, s] == 0)
                            model.addConstr(b[w1, a1, w2, a2, s] == 0)
                        else:
                            model.addConstr(b[w1, a1, w2, a2, s] + b[w2, a2, w1, a1, s] <= 1)
                            model.addConstr(S[w2, a2, s] >= C[w1, a1, s] - M * (1 - b[w1, a1, w2, a2, s]))
                            shared_region = gp.quicksum(z[w1, a1, r, s] * z[w2, a2, r, s] for r in regions)
                            model.addConstr(o[w1, a1, w2, a2, s] >= shared_region - b[w1, a1, w2, a2, s] - b[w2, a2, w1, a1, s])
                            model.addConstr(o[w1, a1, w2, a2, s] <= shared_region)
                            model.addConstr(o[w1, a1, w2, a2, s] <= 1 - b[w1, a1, w2, a2, s])
                            model.addConstr(o[w1, a1, w2, a2, s] <= 1 - b[w2, a2, w1, a1, s])

        for w in workflows:
            for a in roles:
                n_overlap = gp.quicksum(o[w, a, w2, a2, s] for w2 in workflows for a2 in roles if (w2, a2) != (w, a))
                model.addConstr(n_overlap == 1 * is_ov1[w, a, s] + 2 * is_ov2[w, a, s])
                model.addConstr(is_ov1[w, a, s] + is_ov2[w, a, s] <= 1)

        # 场景级别 KV Cache 注意力可见性复用
        for w in workflows:
            for (u, v) in kv_savings.keys():
                same_reg = gp.quicksum(z[w, u, r, s] * z[w, v, r, s] for r in regions)
                model.addConstr(share_kv[w, u, v, s] == same_reg)
                model.addConstr(active_save[w, u, v, s] <= share_kv[w, u, v, s])
                
                if (u, v) in mas_edges or (u, v) in [("A", "C"), ("A", "D"), ("A", "E"), ("B", "E"), ("C", "E"), ("D", "E")]:
                    pass
                elif (u, v) == ("C", "D") or (u, v) == ("D", "C"):
                    model.addConstr(active_save[w, u, v, s] <= b[w, u, w, v, s])
            
            for target_a in roles:
                valid_srcs = [src for (src, tgt) in kv_savings.keys() if tgt == target_a]
                if valid_srcs:
                    model.addConstr(gp.quicksum(active_save[w, src, target_a, s] for src in valid_srcs) <= 1)

        # 动态流水推演与多场景路由延迟
        for w in workflows:
            def trans_d(u, v): return gp.quicksum(z[w, u, r1, s] * z[w, v, r2, s] * get_trans_delay(s, r1, r2) for r1 in regions for r2 in regions)
            
            for a in roles:
                base_t = base_time[a]
                penalty = 0.25 * base_t * is_ov1[w, a, s] + 1.0 * base_t * is_ov2[w, a, s]
                saved = gp.quicksum(active_save[w, src, a, s] * kv_savings[(src, a)] for src in [s_p for (s_p, t) in kv_savings.keys() if t == a])
                
                model.addConstr(D_var[w, a, s] == base_t + penalty - saved)
                model.addConstr(D_var[w, a, s] >= 0.1) 
                model.addConstr(C[w, a, s] == S[w, a, s] + D_var[w, a, s])

            model.addConstr(S[w, "A", s] >= 0)
            for u, v in mas_edges:
                model.addConstr(S[w, v, s] >= C[w, u, s] + trans_d(u, v))
                
            model.addConstr(T_max_s[s] >= C[w, "E", s])

        # CVaR 尾部分布线性化约束
        model.addConstr(u_vars[s] >= T_max_s[s] - eta)

    # 建立多场景宏观期望目标
    cost_s = {}
    for s in scenarios:
        cost_elec = gp.quicksum(z[w, role, r, s] * elec_cost[r] for w in workflows for role in roles for r in regions)
        cost_bw = gp.quicksum(z[w, u, r1, s] * z[w, v, r2, s] * get_bw_cost(s, r1, r2) 
                              for w in workflows for (u,v) in mas_edges for r1 in regions for r2 in regions)
        cost_s[s] = cost_elec + cost_bw
        
    model.addConstr(Expected_Cost == gp.quicksum(prob[s] * cost_s[s] for s in scenarios))
    model.addConstr(CVaR_T == eta + (1.0 / (1.0 - alpha_cvar)) * gp.quicksum(prob[s] * u_vars[s] for s in scenarios))

    # ---------------- 4. 优化求解与帕累托前沿动态步进 ----------------
    print(">>> 探测 CVaR 系统极限边界...")
    model.setObjective(CVaR_T, GRB.MINIMIZE)
    model.optimize()
    if model.status != GRB.OPTIMAL: 
        print("模型无解。")
        return
    T_min, C_max = CVaR_T.X, Expected_Cost.X
    
    model.setObjective(Expected_Cost, GRB.MINIMIZE)
    model.optimize()
    C_min, T_max_val = Expected_Cost.X, CVaR_T.X

    print(f"极小 CVaR 时间点: [CVaR_Time: {T_min:.2f}, Expected_Cost: {C_max:.2f}]")
    print(f"极小期望成本点: [CVaR_Time: {T_max_val:.2f}, Expected_Cost: {C_min:.2f}]")

    eps_constr = model.addConstr(Expected_Cost <= C_max, name="eps_constraint")
    model.setObjective(CVaR_T + 1e-4 * Expected_Cost, GRB.MINIMIZE)
    
    pareto_points = []
    seen_solutions = set()
    current_cost_limit = C_max
    search_count = 0
    max_solutions = 100

    print("\n>>> 沿着帕累托风险前沿进行动态步进搜索...")
    while search_count < max_solutions and current_cost_limit >= C_min:
        eps_constr.RHS = current_cost_limit
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            t_val = round(CVaR_T.X, 3)
            c_val = round(Expected_Cost.X, 3)
            
            if (t_val, c_val) not in seen_solutions:
                seen_solutions.add((t_val, c_val))
                search_count += 1
                
                sol = {
                    'ID': search_count,
                    'Total_Time': t_val, # 此处代表 CVaR_T
                    'Total_Cost': c_val, # 此处代表 Expected_Cost
                    'Deployment': {},
                    'Workflows': {'W1': {'Paths': []}, 'W2': {'Paths': []}}
                }
                
                # 第一阶段决策结果（物理位置是固定的）
                for i in all_instances:
                    for r in regions:
                        if d[i, r].X > 0.5: sol['Deployment'][i] = r
                
                # 第二阶段决策输出（由于路径跟场景挂钩，默认抓取正常场景 s0 的实际调用链路供展示）
                for w in workflows:
                    active_insts = [i for i in all_instances if x[w, i, "s0"].X > 0.5]
                    inst_map = {inst.split('_')[0]: inst for inst in active_insts}
                    for u, v in mas_edges:
                        sol['Workflows'][w]['Paths'].append({
                            'from_agent': inst_map[u],
                            'to_agent': inst_map[v]
                        })
                
                pareto_points.append(sol)
                print(f"发现非支配解 #{search_count}: CVaR_Time={t_val:.2f}, Expected_Cost={c_val:.2f}")

            current_cost_limit = c_val - 0.1 
        else:
            break

    if not pareto_points:
        print("未找到有效解。")
        return

    # 数据集导出
    best_solution = pareto_points[len(pareto_points)//2]
    with open("pareto_solutions.json", "w", encoding='utf-8') as f:
        json.dump(pareto_points, f, indent=4, ensure_ascii=False)
        
    with open("pareto_front_details.md", "w", encoding='utf-8') as f:
        f.write("# MAS 多目标帕累托前沿解集 (CVaR 弹性鲁棒版)\n\n")
        f.write(f"共发现 **{len(pareto_points)}** 组本质不同的非支配解。\n\n")
        for p in pareto_points:
            f.write(f"## 解 ID: {p['ID']}\n")
            f.write(f"- **最大风险完工时间 (CVaR Time)**: {p['Total_Time']} 秒\n")
            f.write(f"- **综合数学期望成本 (Expected Cost)**: {p['Total_Cost']}\n")
            f.write(f"- **15个实例离线部署分布 (一阶段决策)**:\n")
            for inst, r in sorted(p['Deployment'].items()):
                f.write(f"  - {inst} -> {r}\n")
            f.write(f"- **正常无故障状态下 (s0) 的任务流传输调用链路**:\n")
            for w in workflows:
                f.write(f"  - **{w}**:\n")
                for path in p['Workflows'][w]['Paths']:
                    f.write(f"    - {path['from_agent']} -> {path['to_agent']}\n")
            f.write("\n---\n")

    print(f"\n搜索结束，已捕获 {len(pareto_points)} 组非支配解，相关文件已导出。")

    # ---------------- 5. 拓扑与链路可视化 (渲染 s0 正常形态) ----------------
    region_base_pos = {
        "R1": (0, 4), "R2": (4, 4), "R3": (4, 0), "R4": (0, 0), 
        "R5": (1, 3), "R6": (3, 3), "R7": (3, 1), "R8": (1, 1)  
    }
    
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # 绘制基础物理链路
    for u, v in physical_edges:
        pu, pv = region_base_pos[u], region_base_pos[v]
        plt.plot([pu[0], pv[0]], [pu[1], pv[1]], color='lightgray', linestyle='-', linewidth=2, zorder=0)
        mid_x, mid_y = (pu[0] + pv[0]) / 2, (pu[1] + pv[1]) / 2
        cost = edge_costs.get((u, v), edge_costs.get((v, u), 20.0))
        plt.text(mid_x, mid_y, str(int(cost)), color='gray', fontsize=10, 
                 ha='center', va='center', bbox=dict(facecolor='white', edgecolor='gray', pad=2), zorder=1)

    for r, (rx, ry) in region_base_pos.items():
        circle = plt.Circle((rx, ry), 0.45, color='whitesmoke', ec='gainsboro', alpha=0.8, zorder=2)
        ax.add_patch(circle)
        plt.text(rx, ry + 0.55, r, fontsize=12, fontweight='bold', ha='center', color='black')

    agent_pos = {}
    for r, (rx, ry) in region_base_pos.items():
        agents_in_r = [inst for inst, reg in best_solution['Deployment'].items() if reg == r]
        n = len(agents_in_r)
        for idx, inst in enumerate(agents_in_r):
            if n == 1: offset_x, offset_y = 0, 0
            else: offset_x, offset_y = -0.2 + idx*0.4, 0
            
            agent_pos[inst] = (rx + offset_x, ry + offset_y)
            plt.plot(agent_pos[inst][0], agent_pos[inst][1], marker='o', markersize=12, 
                     color='skyblue', markeredgecolor='black', zorder=3)
            plt.text(agent_pos[inst][0], agent_pos[inst][1]-0.2, inst, fontsize=8, ha='center', zorder=4)

    colors = {'W1': 'red', 'W2': 'blue'}
    for w in workflows:
        paths = best_solution['Workflows'][w]['Paths']
        for path in paths:
            u_inst, v_inst = path['from_agent'], path['to_agent']
            pos_u, pos_v = agent_pos[u_inst], agent_pos[v_inst]
            
            plt.annotate("", xy=pos_v, xytext=pos_u,
                         arrowprops=dict(arrowstyle="->", color=colors[w], 
                                         lw=1.8, shrinkA=10, shrinkB=10, 
                                         connectionstyle="arc3,rad=0.2"), zorder=5)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lightgray', lw=2, label='Bandwidth Links'),
        Line2D([0], [0], color='red', lw=2, label='W1 (Normal s0)'),
        Line2D([0], [0], color='blue', lw=2, label='W2 (Normal s0)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.title(f"Instance-Level Elastic Routing (Trade-off Front Solution ID: {best_solution['ID']})")
    plt.axis('equal')
    plt.axis('off')
    plt.savefig("optimal_routing_instances.png", dpi=300, bbox_inches='tight')
    print("已成功更新鲁棒性拓扑渲染图: optimal_routing_instances.png")

if __name__ == "__main__":
    run_pareto_mas_optimization()