use std::collections::{HashMap, HashSet};

use super::{Distance, VertexId};
mod state;
use state::{DfsEvent, LRState, Time};

impl Distance {

pub fn is_planar(&self) -> bool {
    let state = &mut LRState::new(self);

    let time = &mut 0;
    let discovered = &mut HashSet::with_capacity(self.order());
    let finished = &mut HashSet::with_capacity(self.order());

    // DFS orientation phase
    for node in self.vertices() {
        dfs_visitor(self.clone(), node, state, discovered, finished, time);
    }

    // L-R partition phase

    for v in state.roots.clone() {
        // let res = lr_visit_ordered_dfs_tree(&mut state, v, |state, event| {
        //     state.lr_testing_visitor(event)
        // });
        // if res.is_err() {
        //     return false;
        // }
    }


    false
}
}

/// Control flow for `depth_first_search` callbacks.
#[derive(Copy, Clone, Debug)]
pub enum Control<B> {
    /// Continue the DFS traversal as normal.
    Continue,
    /// Prune the current node from the DFS traversal. No more edges from this
    /// node will be reported to the callback. A `DfsEvent::Finish` for this
    /// node will still be reported. This can be returned in response to any
    /// `DfsEvent`, except `Finish`, which will panic.
    Prune,
    /// Stop the DFS traversal and return the provided value.
    Break(B),
}

fn dfs_visitor(
     graph: Distance,
    u: VertexId,
    state: &mut LRState,
    //visitor: &mut F,
    //state.lr_orientation_visitor(event)
    discovered: &mut HashSet<VertexId>,
    finished: &mut HashSet<VertexId>,
    time: &mut Time,
) -> Control<usize> {
    if !discovered.insert(u) {
        return Control::Continue;
    }
    state.lr_orientation_visitor(DfsEvent::Discover(u, time_post_inc(time)));

    let mut stack: Vec<(VertexId, Vec<VertexId>)> = Vec::new();
    stack.push((u, graph.neighbors(u)));

    while let Some((u, neighbors)) = stack.last() {
        let mut next = None;
        for &v in neighbors {
            // is_visited
            if !discovered.contains(&v) {
                //try_control!(visitor(), continue);
                state.lr_orientation_visitor(DfsEvent::TreeEdge(*u, v));
                discovered.insert(v);
                // try_control!(
                state.lr_orientation_visitor(DfsEvent::Discover(v, time_post_inc(time)));
                //     continue
                // );
                next = Some(v);
                break;
            } else if !finished.contains(&v) {
                // try_control!(
                state.lr_orientation_visitor(DfsEvent::BackEdge(*u, v));
                // , continue);
            } else {
                // try_control!(
                state.lr_orientation_visitor(DfsEvent::CrossForwardEdge(*u, v));
                //     continue
                // );
            }
        }

        match next {
            Some(v) => stack.push((v, graph.neighbors(v))),
            None => {
                let first_finish = finished.insert(*u);
                debug_assert!(first_finish);
                state.lr_orientation_visitor(DfsEvent::Finish(*u, time_post_inc(time)));
                //panic!("Pruning on the `DfsEvent::Finish` is not supported!")
                stack.pop();
            }
        };
    }

    Control::Continue
}

fn time_post_inc(x: &mut Time) -> Time {
    let v = *x;
    *x += 1;
    v
}


/// Visits the DFS - oriented tree that we have pre-computed
/// and stored in ``lr_state``. We traverse the edges of
/// a node in nesting depth order. Events are emitted at points
/// of interest and should be handled by ``visitor``.
fn lr_visit_ordered_dfs_tree<G, F, E>(
    lr_state: &mut LRState,
    v: VertexId,
    // mut visitor: F,
) -> Result<(), E>
// where
//     G: GraphBase + IntoEdges,
//     G::NodeId: Hash + Eq,
//     F: FnMut(&mut LRState<G>, LRTestDfsEvent<G::NodeId>) -> Result<(), E>,
{
    let mut stack: Vec<(VertexId, IntoIter<Edge<G>>)> = vec![(
        v,
        edges_filtered_and_sorted_by(
            lr_state.graph,
            v,
            // if ``lowpt`` does *not* contain edge ``e = (v, w)``, it means
            // that it's *not* a tree or a back edge so we skip it since
            // it's oriented in the reverse direction.
            |e| lr_state.lowpt.contains_key(e),
            // we sort edges based on nesting depth order.
            |e| lr_state.nesting_depth[e],
        ),
    )];

    while let Some(elem) = stack.last_mut() {
        let v = elem.0;
        let adjacent_edges = &mut elem.1;
        let mut next = None;

        for (v, w) in adjacent_edges {
            if Some(&(v, w)) == lr_state.eparent.get(&w) {
                // tree edge
                visitor(lr_state, LRTestDfsEvent::TreeEdge(v, w))?;
                next = Some(w);
                break;
            } else {
                // back edge
                visitor(lr_state, LRTestDfsEvent::BackEdge(v, w))?;
                visitor(lr_state, LRTestDfsEvent::FinishEdge(v, w))?;
            }
        }

        match next {
            Some(w) => stack.push((
                w,
                edges_filtered_and_sorted_by(
                    lr_state.graph,
                    w,
                    |e| lr_state.lowpt.contains_key(e),
                    |e| lr_state.nesting_depth[e],
                ),
            )),
            None => {
                stack.pop();
                visitor(lr_state, LRTestDfsEvent::Finish(v))?;
                if let Some(&(u, v)) = lr_state.eparent.get(&v) {
                    visitor(lr_state, LRTestDfsEvent::FinishEdge(u, v))?;
                }
            }
        }
    }

    Ok(())
}
