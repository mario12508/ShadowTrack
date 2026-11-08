"""Global maximum-weight association with MILP and lazy path conflict cuts.

Unlike nearest-neighbour or greedy unions, each iteration optimizes all eligible
edges together. Path cuts prohibit whole transitive chains between cannot-links.
The result carries solver gap/status. A timed-out non-feasible partition is never
silently accepted: unsupported edges are left unresolved.
"""
import time
from collections import Counter,defaultdict,deque
from itertools import combinations

import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import lil_matrix


def global_partition(tracks,candidates,scores,config):
    from .mtmc import topology
    index={t['id']:i for i,t in enumerate(tracks)};n=len(tracks)
    edges=[p for p in candidates if p['status']=='ELIGIBLE'];cannot={};appearance_costs={}
    duplicates={tuple(sorted((index[p['track_a']],index[p['track_b']]))) for p in candidates if p['evidence'].get('partial_duplicate')}
    for i,j in combinations(range(n),2):
        a,b=tracks[i],tracks[j];overlap=min(a['aligned_end'],b['aligned_end'])-max(a['aligned_start'],b['aligned_start'])
        reason=None;path=topology(config,a['camera_id'],b['camera_id'])
        clock_budget=(a['time_uncertainty']+b['time_uncertainty']) if config.get('mode')=='supported_alignment' else config['clock_tolerance_seconds']
        if a['camera_id']==b['camera_id'] and overlap>.001 and (i,j) not in duplicates:reason='simultaneous_same_camera_conflict'
        elif a['camera_id']!=b['camera_id'] and path and not path['overlap_possible'] and overlap>clock_budget:reason='component_topology_conflict'
        elif scores[i,j]>=0 and scores[i,j]<config['component_minimum_similarity']:
            # A front/back disagreement is negative evidence, not a physical
            # impossibility. Charge it to the WHOLE component rather than
            # vetoing a chain supported by independent observed motion.
            # Nested duplicates share one descriptor, hence one total cost.
            size=max(1,a.get('local_bundle_size',1))*max(1,b.get('local_bundle_size',1))
            from .association_features import clothing_similarity
            clothing=clothing_similarity(a.get('color_histogram'),b.get('color_histogram'))
            # Agreement of a separately computed clothing histogram weakens a
            # moderate cross-view descriptor objection; never erase it, and
            # never discount a severe appearance contradiction (< .55).
            reliability=max(.35,1-clothing) if clothing is not None and a['camera_id']!=b['camera_id'] and scores[i,j]>=.55 else 1.
            appearance_costs[i,j]=(config['component_minimum_similarity']-scores[i,j])*reliability/size
        if reason:cannot[i,j]=reason
    for p in candidates:
        if 'human_rejected' in p['reasons']:cannot[tuple(sorted((index[p['track_a']],index[p['track_b']])))]='human_rejected'
    # Competing lookalikes must be considered concurrently; don't collapse a
    # soft nearest-neighbour tie into a positive identity assertion.
    for p in edges:
        i,j=index[p['track_a']],index[p['track_b']]
        for q in edges:
            if q is p or abs(q['score']-p['score'])>=config['ambiguity_margin']:continue
            common=set((p['track_a'],p['track_b']))&set((q['track_a'],q['track_b']))
            if not common:continue
            shared=next(iter(common));x=index[next(k for k in (p['track_a'],p['track_b']) if k!=shared)];y=index[next(k for k in (q['track_a'],q['track_b']) if k!=shared)]
            if tracks[x]['camera_id']==tracks[y]['camera_id'] and tuple(sorted((x,y))) not in duplicates and max(tracks[x]['aligned_start'],tracks[y]['aligned_start'])<min(tracks[x]['aligned_end'],tracks[y]['aligned_end']):
                p['status']='UNRESOLVED';p['reasons'].append('competing_simultaneous_lookalikes');break
    edges=[p for p in edges if p['status']=='ELIGIBLE'];endpoints=[(index[p['track_a']],index[p['track_b']]) for p in edges]
    weights=np.asarray([max(.001,p['score']-config['appearance_threshold']+.02) for p in edges])
    def bundle_pair(k):
        a,b=tracks[endpoints[k][0]],tracks[endpoints[k][1]]
        return tuple(sorted((a.get('local_bundle_id') or a['id'],b.get('local_bundle_id') or b['id'])))
    multiplicity=Counter(bundle_pair(k) for k in range(len(edges)))
    for k,p in enumerate(edges):
        a,b=tracks[endpoints[k][0]],tracks[endpoints[k][1]]
        same_bundle=a.get('local_bundle_id') and a.get('local_bundle_id')==b.get('local_bundle_id')
        # Three nested boxes carrying one pooled descriptor are not three
        # independent cross-camera votes. Count their total reward once too.
        if not same_bundle:weights[k]/=multiplicity[bundle_pair(k)]
        review=p['evidence'].get('human_review')
        if review and review['decision']=='CONFIRM':weights[k]+=10
    cuts=[];cut_set=set();started=time.monotonic();chosen=set();solver_status='EMPTY';solver_gap=None
    def components(active):
        graph=defaultdict(list)
        for k in active:
            i,j=endpoints[k];graph[i].append((j,k));graph[j].append((i,k))
        groups=[];visited=set()
        for i in range(n):
            if i in visited:continue
            group=[];todo=[i];visited.add(i)
            while todo:
                u=todo.pop();group.append(u)
                for v,k in graph[u]:
                    if v not in visited:visited.add(v);todo.append(v)
            groups.append(group)
        return groups,graph
    possible_groups,_=components(range(len(edges)))
    possible={i:k for k,group in enumerate(possible_groups) for i in group}
    penalties={pair:cost for pair,cost in appearance_costs.items() if possible[pair[0]]==possible[pair[1]]}
    penalty_indexes={pair:len(edges)+k for k,pair in enumerate(sorted(penalties))}
    objective=np.r_[-weights,[penalties[pair] for pair in sorted(penalties)]]
    def conflicts(active,paid=(),include_soft=True):
        groups,graph=components(active);violations=[]
        for group in groups:
            for a,b in combinations(sorted(group),2):
                hard=(a,b) in cannot;soft=include_soft and (a,b) in penalties and (a,b) not in paid
                if not hard and not soft:continue
                queue=deque([a]);previous={a:None}
                while queue and b not in previous:
                    u=queue.popleft()
                    for v,k in graph[u]:
                        if v not in previous:previous[v]=(u,k);queue.append(v)
                path=[];node=b
                while node!=a:node,k=previous[node];path.append(k)
                violations.append((tuple(sorted(path)),None if hard else penalty_indexes[a,b]))
        return groups,violations
    if edges:
        for iteration in range(32):
            remaining=max(.1,config.get('assignment_time_limit_seconds',12)-(time.monotonic()-started))
            matrix=lil_matrix((len(cuts),len(objective)))
            for r,(cut,penalty) in enumerate(cuts):
                matrix[r,list(cut)]=1
                if penalty is not None:matrix[r,penalty]=-1
            constraints=LinearConstraint(matrix.tocsr(),-np.inf,np.asarray([len(cut)-1 for cut,_ in cuts])) if cuts else None
            solution=milp(objective,integrality=np.ones(len(objective)),bounds=Bounds(0,1),constraints=constraints,options={'time_limit':remaining,'mip_rel_gap':.001})
            solver_status=str(solution.message);solver_gap=float(solution.mip_gap) if getattr(solution,'mip_gap',None) is not None else None
            if solution.x is None:chosen=set();break
            active={i for i,x in enumerate(solution.x[:len(edges)]) if x>.5}
            paid={pair for pair,k in penalty_indexes.items() if solution.x[k]>.5};groups,violations=conflicts(active,paid)
            if not violations:chosen=active;break
            for cut,penalty in violations:
                key=(cut,penalty)
                if key not in cut_set:cuts.append(key);cut_set.add(key)
            if remaining<=.2 or iteration==31:
                # Safe bounded fallback: keep only conflict-free solved components.
                chosen=set(active)
                for cut,_ in violations:chosen.difference_update(cut)
                while True:
                    _,remaining_conflicts=conflicts(chosen,include_soft=False)
                    if not remaining_conflicts:break
                    for cut,_ in remaining_conflicts:chosen.difference_update(cut)
                solver_status+='; conflicting components left unresolved';break
        for k,p in enumerate(edges):
            if k in chosen:
                review=p['evidence'].get('human_review');p['status']='CONFIRMED' if review and review['decision']=='CONFIRM' else 'PROVISIONAL'
            else:p['status']='UNRESOLVED';p['reasons'].append('global_optimization_conflict_or_alternative')
            p['evidence']['global_solver']={'method':'MILP positive links and component appearance costs; hard physical path cuts','status':solver_status,'mip_gap':solver_gap,'cuts':len(cuts)}
    groups,_=components(chosen)
    membership={i:k for k,group in enumerate(groups) for i in group}
    paid_cost=sum(cost for (i,j),cost in penalties.items() if membership[i]==membership[j])
    return groups,{'method':'MILP_GRAPH_MULTICUT','edges':len(edges),'cuts':len(cuts),'appearance_penalty_pairs':len(penalties),'component_appearance_cost':float(paid_cost),'status':solver_status,'mip_gap':solver_gap,'wall_seconds':time.monotonic()-started}
