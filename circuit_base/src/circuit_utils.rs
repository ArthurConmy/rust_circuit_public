use std::{collections::BTreeSet, hash::Hash, iter::zip};

use num_bigint::BigUint;
use pyo3::prelude::*;
use rr_util::{
    name::Name,
    opt_einsum::EinsumSpec,
    tensor_util::{TensorAxisIndex, TensorIndex, TorchDeviceDtype, TorchDtype},
};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::{
    apply_fn_cache, deep_map_op_context_preorder_stoppable, deep_map_op_pre_new_children,
    deep_map_unwrap, visit_circuit_non_free, visit_circuit_unwrap, Add, Array, Circuit,
    CircuitNode, CircuitNodeAutoName, CircuitRc, Concat, Conv, Cumulant, DiscreteVar, Einsum,
    GeneralFunction, HashBytes, Index, Module, Rearrange, Scatter, SetSymbolicShape,
    StoredCumulantVar, Tag,
};

#[pyfunction]
pub fn cast_circuit(
    circ: CircuitRc,
    device_dtype: &rr_util::tensor_util::TorchDeviceDtypeOp,
) -> CircuitRc {
    deep_map_op_pre_new_children(
        circ.clone(),
        |circ, children: &Vec<CircuitRc>| match &**circ {
            Circuit::Array(node) => {
                if device_dtype
                    .clone()
                    .combine(TorchDeviceDtype::from_tensor(&node.value).into())
                    .is_err()
                {
                    Some(Array::nrc(
                        device_dtype.cast_tensor(node.value.clone()),
                        node.info().name,
                    ))
                } else {
                    None
                }
            }
            Circuit::Index(node) => {
                if node
                    .index
                    .0
                    .iter()
                    .any(|x| matches!(x, TensorAxisIndex::Tensor(_)))
                {
                    Some(Index::nrc(
                        children[0].clone(),
                        TensorIndex(
                            node.index
                                .0
                                .iter()
                                .map(|idx| match idx {
                                    TensorAxisIndex::Tensor(tensor) => TensorAxisIndex::Tensor(
                                        TorchDeviceDtype {
                                            device: device_dtype
                                                .clone()
                                                .device
                                                .unwrap_or(
                                                    TorchDeviceDtype::from_tensor(&tensor).device,
                                                )
                                                .clone(),
                                            dtype: TorchDtype::int64,
                                        }
                                        .cast_tensor((**tensor).clone())
                                        .try_into()
                                        .unwrap(),
                                    ),
                                    _ => idx.clone(),
                                })
                                .collect(),
                        ),
                        node.info().name,
                    ))
                } else {
                    None
                }
            }
            _ => None,
        },
    )
    .unwrap_or(circ)
}

#[pyfunction]
pub fn count_nodes(circuit: CircuitRc) -> usize {
    let mut result: usize = 0;
    visit_circuit_unwrap(circuit, &mut |_x: CircuitRc| {
        result += 1;
    });
    result
}

pub fn hash_to_node_non_free(
    circuit: CircuitRc,
    recur_into_free: bool,
) -> HashMap<HashBytes, CircuitRc> {
    let mut result: HashMap<HashBytes, CircuitRc> = HashMap::default();
    visit_circuit_non_free(
        circuit,
        &mut |x: CircuitRc| {
            result.insert(x.info().hash, x.clone());
            Ok(())
        },
        recur_into_free,
    )
    .unwrap();
    result
}

pub fn hash_to_node(circuit: CircuitRc) -> HashMap<HashBytes, CircuitRc> {
    hash_to_node_non_free(circuit, true)
}

pub fn subcircuits(circuit: CircuitRc) -> HashSet<CircuitRc> {
    let mut result: HashSet<CircuitRc> = HashSet::default();
    visit_circuit_unwrap(circuit, &mut |x: CircuitRc| {
        result.insert(x);
    });
    result
}

#[pyfunction]
pub fn total_flops(circuit: CircuitRc) -> BigUint {
    let mut result: BigUint = BigUint::from(0usize);
    visit_circuit_unwrap(circuit, &mut |x: CircuitRc| {
        result += x.self_flops();
    });
    result
}

pub fn total_flops_cached(circuit: CircuitRc, cache: &mut HashMap<HashBytes, BigUint>) -> BigUint {
    apply_fn_cache(
        &circuit,
        |x| total_flops(x.clone()),
        cache,
        |x| x.info().hash,
    )
}

#[pyfunction]
pub fn total_arrayconstant_size(circuit: CircuitRc) -> BigUint {
    let mut result: BigUint = BigUint::from(0usize);
    visit_circuit_unwrap(circuit, &mut |x: CircuitRc| {
        if let Circuit::Array(x) = &**x {
            result += x.info().numel();
        }
    });
    result
}

#[pyfunction]
pub fn sum_of_node_sizes(circuit: CircuitRc) -> BigUint {
    let mut result: BigUint = BigUint::from(0usize);
    visit_circuit_unwrap(circuit, &mut |x: CircuitRc| {
        if !matches!(&**x, Circuit::Array(_)) {
            result += x.info().numel();
        }
    });
    result
}

pub fn sum_of_node_sizes_cached(
    circuit: CircuitRc,
    cache: &mut HashMap<HashBytes, BigUint>,
) -> BigUint {
    apply_fn_cache(
        &circuit,
        |x| sum_of_node_sizes(x.clone()),
        cache,
        |x| x.info().hash,
    )
}

#[pyfunction]
pub fn get_leaves(circuit: CircuitRc) -> Vec<CircuitRc> {
    let mut result: Vec<CircuitRc> = vec![];
    visit_circuit_unwrap(circuit, &mut |c: CircuitRc| {
        if c.num_children() == 0 {
            result.push(c);
        }
    });
    result
}

#[pyfunction]
pub fn get_all_einsum_specs(circuit: CircuitRc) -> Vec<EinsumSpec> {
    let mut result: Vec<EinsumSpec> = vec![];
    visit_circuit_unwrap(circuit, |c| {
        if let Circuit::Einsum(node) = &**c {
            result.push(node.get_spec());
        }
    });
    result
}

#[pyfunction]
/// children first
pub fn toposort_circuit(circuit: CircuitRc) -> Vec<CircuitRc> {
    let mut num_refs: HashMap<CircuitRc, usize> = HashMap::default();
    visit_circuit_unwrap(circuit.clone(), |c| {
        for child in c.children() {
            *num_refs.entry(child).or_insert(0) += 1;
        }
    });
    let mut ready: BTreeSet<CircuitRc> = BTreeSet::from([circuit]);
    let mut result: Vec<CircuitRc> = vec![];
    while !ready.is_empty() {
        let here = ready.pop_first().unwrap();
        for child in here.children() {
            num_refs.insert(child.clone(), num_refs[&child] - 1);
            if num_refs[&child] == 0 {
                ready.insert(child);
            }
        }
        result.push(here.clone())
    }
    result.reverse();
    result
}

pub fn deep_map_pass_up<F, T>(circuit: CircuitRc, mut f: F) -> (CircuitRc, T)
where
    T: Clone,
    F: FnMut(CircuitRc, Vec<T>) -> (CircuitRc, T),
{
    let topo = toposort_circuit(circuit.clone());
    let mut pass_ups: HashMap<CircuitRc, T> = HashMap::default();
    for c in topo {
        let passed_up = c.children().map(|c| pass_ups[&c].clone()).collect();
        let (new, pass_here) = f(c.clone(), passed_up);
        pass_ups.insert(c.clone(), pass_here.clone());
        if c == circuit {
            return (new, pass_here);
        }
    }
    panic!();
}

pub fn deep_map_pass_down<F, F2, T>(circuit: CircuitRc, pass_down: F, make_circuit: F2) -> CircuitRc
where
    F: Fn(CircuitRc, &Vec<T>) -> Vec<T>,
    F2: Fn(CircuitRc, &Vec<T>) -> CircuitRc,
{
    let mut toposorted = toposort_circuit(circuit.clone());
    toposorted.reverse();
    let mut passed_down: HashMap<CircuitRc, Vec<T>> = HashMap::default();
    for c in toposorted {
        let pass = pass_down(c.clone(), passed_down.get(&c).unwrap_or(&vec![]));
        for (child, pass) in zip(c.children(), pass) {
            passed_down
                .entry(child.clone())
                .or_insert(vec![])
                .push(pass);
        }
    }
    deep_map_unwrap(circuit, |c| {
        make_circuit(c.clone(), passed_down.get(&c).unwrap_or(&vec![]))
    })
}

pub fn deep_map_pass_down_branching<F, F2, T>(
    circuit: CircuitRc,
    pass_down_f: F,
    make_circuit_f: F2,
    initial_pass_down: T,
) -> CircuitRc
where
    T: Hash + Eq + Clone,
    F: Fn(CircuitRc, &T) -> Vec<T>,
    F2: Fn(CircuitRc, &T, &Vec<CircuitRc>) -> CircuitRc,
{
    let mut pass_down_cache: HashMap<(CircuitRc, T), Vec<T>> = HashMap::default();
    let mut construct_cache: HashMap<(CircuitRc, T), CircuitRc> = HashMap::default();

    fn recurse<F, F2, T>(
        circuit: CircuitRc,
        passed: T,
        pdf: &F,
        mcf: &F2,
        pdc: &mut HashMap<(CircuitRc, T), Vec<T>>,
        cc: &mut HashMap<(CircuitRc, T), CircuitRc>,
    ) -> CircuitRc
    where
        T: Hash + Eq + Clone,
        F: Fn(CircuitRc, &T) -> Vec<T>,
        F2: Fn(CircuitRc, &T, &Vec<CircuitRc>) -> CircuitRc,
    {
        let pass_key = (circuit.clone(), passed.clone());
        if let Some(result) = cc.get(&pass_key) {
            return result.clone();
        }
        let passing = pdc.get(&pass_key).cloned().unwrap_or_else(|| {
            let result = pdf(circuit.clone(), &passed);
            pdc.insert(pass_key.clone(), result.clone());
            result
        });
        let new_children: Vec<CircuitRc> = zip(circuit.children(), passing)
            .map(|(child, pass)| recurse(child, pass, pdf, mcf, pdc, cc))
            .collect();

        let result = mcf(circuit, &passed, &new_children);
        cc.insert(pass_key, result.clone());
        result
    }
    recurse(
        circuit,
        initial_pass_down,
        &pass_down_f,
        &make_circuit_f,
        &mut pass_down_cache,
        &mut construct_cache,
    )
}

#[pyfunction]
pub fn replace_all_randn_seeded(circuit: CircuitRc) -> CircuitRc {
    deep_map_unwrap(circuit, |node| match &**node {
        Circuit::Array(ac) => Array::randn_full(
            ac.info().shape.clone(),
            ac.info().name,
            ac.info().device_dtype.clone(),
            // We use the tensor hash instead of the circuit hash
            Some(ac.value.hash_usize().unwrap()),
        )
        .rc(),
        Circuit::Index(index) => Index::nrc(
            index.node().clone(),
            TensorIndex(
                zip(&index.index.0, index.node().info().shape.clone())
                    .map(|(i, l)| {
                        if let TensorAxisIndex::Tensor(t) = i {
                            return TensorAxisIndex::new_tensor_randint_seeded(
                                t.shape()[0],
                                l,
                                index.info().device_dtype.clone(),
                                t.hash_usize().unwrap(),
                            );
                        }
                        i.clone()
                    })
                    .collect(),
            ),
            index.info().name,
        ),
        _ => node.clone(),
    })
}

/// Replaces child nodes recursively with a mapping
/// In pre-order so that high level nodes are replaced first
/// This function isn't panic safe even if you replace nodes with identically shaped nodes.
pub fn replace_nodes(circuit: CircuitRc, map: &HashMap<HashBytes, CircuitRc>) -> CircuitRc {
    deep_map_op_context_preorder_stoppable(
        circuit.clone(),
        &|x: CircuitRc, _| {
            let rc = x;
            let result = map.get(&rc.info().hash).cloned();
            let stop = result.is_some();
            (result, stop)
        },
        &mut (),
        &mut Default::default(),
    )
    .unwrap_or(circuit)
}

#[pyfunction]
pub fn prefix_all_names(circuit: CircuitRc, prefix: String) -> CircuitRc {
    deep_map_unwrap(circuit, |x| {
        if let Some(name) = x.info().name {
            return x
                .clone()
                .rename(Some((prefix.clone() + name.into()).into()));
        }
        x
    })
}

/// For naming only
#[derive(Debug)]
pub enum OperatorPriority {
    Infix { priority: u8 },
    InfixAmbiguous {},
    PostFix {},
    Function {},
    NotOperator {},
}

pub fn get_priority(circuit: CircuitRc) -> OperatorPriority {
    match &**circuit {
        Circuit::Einsum(_) => Einsum::PRIORITY,
        Circuit::Add(_) => Add::PRIORITY,
        Circuit::Concat(_) => Concat::PRIORITY,
        Circuit::Rearrange(_) => Rearrange::PRIORITY,
        Circuit::Index(_) => Index::PRIORITY,
        Circuit::Scatter(_) => Scatter::PRIORITY,
        Circuit::SetSymbolicShape(_) => SetSymbolicShape::PRIORITY,
        Circuit::Tag(_) => Tag::PRIORITY,
        Circuit::DiscreteVar(_) => DiscreteVar::PRIORITY,
        Circuit::GeneralFunction(_) => GeneralFunction::PRIORITY,
        Circuit::Module(_) => Module::PRIORITY,
        Circuit::Cumulant(_) => Cumulant::PRIORITY,
        Circuit::StoredCumulantVar(_) => StoredCumulantVar::PRIORITY,
        Circuit::Conv(_) => Conv::PRIORITY,
        _ => OperatorPriority::NotOperator {},
    }
}

#[test]
fn f() {
    println!("{:?}", Einsum::PRIORITY)
}

pub fn do_add_parenthesis_to_name(parent_priority: &OperatorPriority, child: CircuitRc) -> bool {
    match (parent_priority, get_priority(child)) {
        (OperatorPriority::NotOperator {}, _) => false, // Should not happen
        (OperatorPriority::Function {}, _) => false,
        (_, OperatorPriority::NotOperator {}) | (_, OperatorPriority::Function {}) => false,
        (OperatorPriority::PostFix {}, OperatorPriority::PostFix {}) => false,
        (OperatorPriority::PostFix {}, _) => true,
        (_, OperatorPriority::PostFix {}) => false,
        (OperatorPriority::InfixAmbiguous {}, _) => true,
        (OperatorPriority::Infix { .. }, OperatorPriority::InfixAmbiguous {}) => true,
        (
            OperatorPriority::Infix { priority: p_parent },
            OperatorPriority::Infix { priority: p_child },
        ) => *p_parent > p_child,
    }
}

/// Return the name surrounded with parenthesis if appropriate, and None if name is None
pub fn child_name_with_maybe_paren(
    parent_priority: &OperatorPriority,
    child: CircuitRc,
) -> Option<Name> {
    child.info().name.map(|name| {
        if do_add_parenthesis_to_name(parent_priority, child) {
            format!("({name})").into()
        } else {
            name
        }
    })
}

/// Return the names surrounded with parenthesis if appropriate, and None if any of the name is None
pub fn children_names_with_maybe_paren(
    parent_priority: &OperatorPriority,
    children: Vec<CircuitRc>,
) -> Option<Vec<Name>> {
    if children.iter().any(|x| x.info().name.is_none()) {
        None
    } else {
        Some(
            children
                .into_iter()
                .map(|x| child_name_with_maybe_paren(parent_priority, x).unwrap())
                .collect::<Vec<Name>>(),
        )
    }
}

pub fn is_definitely_view_on_child(circuit: CircuitRc) -> bool {
    if circuit.num_children() != 1 {
        return false;
    }
    match &**circuit {
        Circuit::Rearrange(rr) => {
            rr.spec.input_ints.iter().all(|x| x.len() < 2)
                && rr.spec.output_ints.iter().all(|x| x.len() < 2)
            // this is too conservative
        }
        Circuit::Index(idx) => !idx
            .index
            .0
            .iter()
            .any(|x| matches!(x, TensorAxisIndex::Tensor(_))),
        _ => false,
    }
}
