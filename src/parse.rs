use std::collections::{HashMap, HashSet};

use pest::{
    iterators::{Pair, Pairs},
    Parser,
};
use petgraph::dot::Dot;

use pest_derive::Parser;
use petgraph::{graph::NodeIndex, Graph};
use std::fmt;

#[derive(Parser)]
#[grammar = "rhdl.pest"]
struct RHDLParser;

pub fn parse_hdl(hdl_text: &str) {
    match RHDLParser::parse(Rule::program, hdl_text) {
        Ok(pairs) => {
            parse_chips(pairs);
        }
        Err(error) => println!("{}", error),
    };
}

fn parse_chips(pairs: Pairs<Rule>) {
    let pairs: Vec<Pair<Rule>> = pairs.collect();
    let chip_signatures: HashMap<&str, (Vec<(&str, u32)>, Vec<(&str, u32)>)> = pairs
        .iter()
        .cloned()
        .map(|pair| get_chip_signature(pair))
        .collect();
    let global_context = GlobalContext { chip_signatures };
    let mut chips: HashMap<&str, ChipContext> = pairs
        .iter()
        .cloned()
        .map(|pair| parse_chip(pair, &global_context))
        .collect();
    let netlist = construct_netlist(&global_context, &mut chips);
    println!("{:?}", Dot::new(&netlist));
}

#[derive(Debug, Clone)]
enum NLConnection {
    Wire,
    FlipFlop,
}

fn construct_netlist<'a>(
    global_context: &GlobalContext<'a>,
    chips: &mut HashMap<&str, ChipContext<'a>>,
) -> Graph<String, NLConnection> {
    let mut netlist = Graph::<String, NLConnection>::new();
    let mut inputs = HashMap::new();
    for (identifier, width) in global_context.chip_signatures["Top"].0.iter() {
        inputs.insert(
            *identifier,
            (0..*width)
                .map(|index| netlist.add_node(format!("Top_{}{}", identifier, index)))
                .collect::<Vec<NodeIndex>>(),
        );
    }
    extend_netlist_with_chip("Top", inputs, global_context, chips, &mut netlist, "");
    netlist
}

fn extend_netlist_with_chip<'a>(
    chip_identifier: &str,
    mut bus_nodes: HashMap<&'a str, Vec<NodeIndex>>,
    global_context: &GlobalContext<'a>,
    chips: &HashMap<&str, ChipContext<'a>>,
    netlist: &mut Graph<String, NLConnection>,
    root_name: &str,
) -> HashMap<&'a str, Vec<NodeIndex>> {
    let chip_data = &chips[chip_identifier];
    let output_identifiers: Vec<&str> = global_context.chip_signatures[chip_identifier]
        .1
        .iter()
        .map(|(identifier, _)| *identifier)
        .collect();
    let non_input_identifiers: Vec<&str> = chips[chip_identifier]
        .bus_connections
        .keys()
        .filter(|identifier| !bus_nodes.contains_key(*identifier))
        .cloned()
        .collect();
    for bus_identifier in chips[chip_identifier].bus_connections.keys() {
        if !bus_nodes.contains_key(bus_identifier) {
            let bus_connections = &chip_data.bus_connections[bus_identifier];
            bus_nodes.entry(bus_identifier).or_insert(
                (0..bus_connections.get_width())
                    .map(|index| {
                        netlist.add_node(format!(
                            "{}{}_{}{}",
                            root_name, chip_identifier, bus_identifier, index
                        ))
                    })
                    .collect::<Vec<NodeIndex>>(),
            );
        }
    }
    let mut chip_instantiation_nodes: Vec<(
        ChipInstantiation<'a>,
        HashMap<&'a str, Vec<NodeIndex>>,
    )> = Vec::new();
    for identifier in non_input_identifiers.iter() {
        let edge_weight = match chip_data.bus_connections[identifier].bus_connection_type {
            BusConnectionType::Combinational => NLConnection::Wire,
            BusConnectionType::Sequential => NLConnection::FlipFlop,
            BusConnectionType::Unspecified => panic!("Unspecified connection type"),
        };
        for (bus_slice, wire_connection) in chip_data.bus_connections[identifier]
            .input_connections
            .iter()
        {
            match bus_slice {
                BusSlice::Index(index) => {
                    let node = bus_nodes[identifier][*index as usize];
                    let input_node = wire_connection.get_expression().extend_netlist_for_index(
                        0,
                        chip_identifier,
                        root_name,
                        &mut bus_nodes,
                        global_context,
                        chips,
                        netlist,
                        &mut chip_instantiation_nodes,
                    );
                    netlist.add_edge(input_node, node, edge_weight.clone());
                }
                BusSlice::Range(start, end) => {
                    for (expression_index, node) in (*start..*end)
                        .map(|index| bus_nodes[identifier][index as usize])
                        .enumerate()
                        .collect::<Vec<_>>()
                        .into_iter()
                    {
                        let input_node = wire_connection.get_expression().extend_netlist_for_index(
                            expression_index as u32,
                            chip_identifier,
                            root_name,
                            &mut bus_nodes,
                            global_context,
                            chips,
                            netlist,
                            &mut chip_instantiation_nodes,
                        );
                        netlist.add_edge(input_node, node, edge_weight.clone());
                    }
                }
                BusSlice::Full => {
                    for (expression_index, node) in (0..chip_data.bus_connections[identifier]
                        .get_width())
                        .map(|index| bus_nodes[identifier][index as usize])
                        .enumerate()
                        .collect::<Vec<_>>()
                        .into_iter()
                    {
                        let input_node = wire_connection.get_expression().extend_netlist_for_index(
                            expression_index as u32,
                            chip_identifier,
                            root_name,
                            &mut bus_nodes,
                            global_context,
                            chips,
                            netlist,
                            &mut chip_instantiation_nodes,
                        );
                        netlist.add_edge(input_node, node, edge_weight.clone());
                    }
                }
            }
        }
    }
    bus_nodes.retain(|identifier, _| output_identifiers.contains(identifier));
    bus_nodes
}

fn parse_chip<'a>(
    pair: Pair<'a, Rule>,
    global_context: &GlobalContext<'a>,
) -> (&'a str, ChipContext<'a>) {
    assert!(pair.as_rule() == Rule::chip_declaration);
    let mut chip_context = ChipContext {
        bus_connections: HashMap::new(),
        requisite_chips: HashSet::new(),
    };
    let mut pairs = pair.into_inner();
    let identifier = pairs.next().unwrap().as_str();
    let (chip_inputs, chip_outputs) = &global_context.chip_signatures[identifier];
    for (chip_input_identifier, width) in chip_inputs {
        chip_context.bus_connections.insert(
            *chip_input_identifier,
            BusConnection {
                input_connections: vec![(BusSlice::Full, WireConnection::Auxiliary)],
                output_connections: Vec::new(),
                bus_connection_type: BusConnectionType::Sequential,
                width: BusWidth::Resolved(*width),
            },
        );
    }
    for (chip_output_identifier, width) in chip_outputs {
        chip_context.bus_connections.insert(
            *chip_output_identifier,
            BusConnection {
                input_connections: Vec::new(),
                output_connections: vec![BusSlice::Full],
                bus_connection_type: BusConnectionType::Unspecified,
                width: BusWidth::Resolved(*width),
            },
        );
    }

    for pair in pairs {
        match pair.as_rule() {
            Rule::assignment => {
                let (lhs, assignment_type, rhs) =
                    parse_assignment(pair, global_context, &mut chip_context);
                let bus_connection = chip_context
                    .bus_connections
                    .entry(lhs.identifier)
                    .or_insert(BusConnection {
                        input_connections: Vec::new(),
                        output_connections: Vec::new(),
                        bus_connection_type: BusConnectionType::Unspecified,
                        width: BusWidth::Unresolved,
                    });
                bus_connection.bus_connection_type = assignment_type;
                bus_connection
                    .input_connections
                    .push((lhs.slice, WireConnection::Local(rhs)));
            }
            Rule::tuple_deconstruction => {
                let tuple_pairs = pair.into_inner();
                let mut lhs = Vec::new();
                let mut assignment_type = BusConnectionType::Unspecified;
                for pair in tuple_pairs {
                    match pair.as_rule() {
                        Rule::bus => lhs.push(Bus::from_pair(pair)),
                        Rule::combinational_assignment => {
                            assignment_type = BusConnectionType::Combinational;
                        }
                        Rule::sequential_assignment => {
                            assignment_type = BusConnectionType::Sequential;
                        }
                        Rule::chip_instantiation => {
                            let chip_instantiation = ChipInstantiation::from_pair(
                                pair,
                                global_context,
                                &mut chip_context,
                            );

                            let chip_outputs = &global_context.chip_signatures
                                [chip_instantiation.chip_identifier]
                                .1;
                            if chip_outputs.len() != lhs.len() {
                                panic!("Mismatched sides on tuple deconstruction");
                            }
                            chip_context.requisite_chips.insert(identifier);
                            for (index, bus) in lhs.iter().enumerate() {
                                let width = chip_outputs[index].1;
                                let expression = Expression {
                                    bitwise_reduction: None,
                                    operators: Vec::new(),
                                    operands: vec![Operand::ChipInstantiation(ChipInstantiation {
                                        output_index: Some(index as u32),
                                        ..chip_instantiation.clone()
                                    })],
                                    width: BusWidth::Resolved(width),
                                };
                                let bus_connection = chip_context
                                    .bus_connections
                                    .entry(bus.identifier)
                                    .or_insert(BusConnection {
                                        input_connections: Vec::new(),
                                        output_connections: Vec::new(),
                                        bus_connection_type: BusConnectionType::Unspecified,
                                        width: BusWidth::Unresolved,
                                    });
                                bus_connection.bus_connection_type = assignment_type.clone();
                                bus_connection
                                    .input_connections
                                    .push((bus.slice.clone(), WireConnection::Local(expression)));
                            }
                        }
                        _ => unreachable!(),
                    }
                }
            }
            Rule::input | Rule::output => (),
            _ => {
                unreachable!()
            }
        }
    }

    let mut width_resolution_queue: Vec<_> = chip_context.bus_connections.keys().cloned().collect();
    let mut unresolved = Vec::new();
    let mut previous_iteration_length = usize::MAX;
    while !width_resolution_queue.is_empty() {
        if width_resolution_queue.len() == previous_iteration_length {
            panic!("Unable to resolve widths for {:?}", width_resolution_queue);
        }
        previous_iteration_length = width_resolution_queue.len();
        for bus_identifier in width_resolution_queue {
            if let Some(width) = chip_context.bus_connections[bus_identifier].try_resolve_width(
                global_context,
                &chip_context,
                &mut HashSet::new(),
            ) {
                chip_context
                    .bus_connections
                    .get_mut(bus_identifier)
                    .unwrap()
                    .width = BusWidth::Resolved(width);
            } else {
                unresolved.push(bus_identifier);
            }
        }
        width_resolution_queue = unresolved;
        unresolved = Vec::new();
    }

    (identifier, chip_context)
}

struct GlobalContext<'a> {
    chip_signatures: HashMap<&'a str, (Vec<(&'a str, u32)>, Vec<(&'a str, u32)>)>,
}

struct ChipContext<'a> {
    bus_connections: HashMap<&'a str, BusConnection<'a>>,
    requisite_chips: HashSet<&'a str>,
}

#[derive(Debug, Clone)]
struct BusConnection<'a> {
    input_connections: Vec<(BusSlice, WireConnection<'a>)>,
    output_connections: Vec<BusSlice>,
    bus_connection_type: BusConnectionType,
    width: BusWidth,
}

impl<'a> BusConnection<'a> {
    fn try_resolve_width(
        &self,
        global_context: &GlobalContext,
        chip_context: &ChipContext<'a>,
        explored_buses: &mut HashSet<&'a str>,
    ) -> Option<u32> {
        if let BusWidth::Resolved(width) = self.width {
            Some(width)
        } else if self.input_connections.len() == 1 {
            let connection = self.input_connections.last().unwrap();
            if connection.0 == BusSlice::Full {
                if let WireConnection::Local(expression) = &connection.1 {
                    expression.try_resolve_width(global_context, chip_context, explored_buses)
                } else {
                    panic!("Auxiliary connection with unresolved width");
                }
            } else {
                panic!("Solitary slice is not full");
            }
        } else if self.input_connections.len() > 1 {
            let mut ranges: Vec<(u32, u32)> = self
                .input_connections
                .iter()
                .map(|(slice, _)| match slice {
                    BusSlice::Full => panic!("Full input slice mixed with other slice"),
                    BusSlice::Index(i) => (*i, i + 1),
                    BusSlice::Range(start, end) => (*start, *end),
                })
                .collect();
            ranges.sort();
            if ranges.windows(2).any(|window| window[0].1 != window[1].0) {
                panic!("Bus slices do not form a contiguous whole")
            }
            if ranges[0].0 != 0 {
                panic!("Ranges do not start at 0")
            }
            Some(ranges.into_iter().map(|(start, end)| end - start).sum())
        } else {
            panic!("Bus has no assignments");
        }
    }

    fn get_width(&self) -> u32 {
        if let BusWidth::Resolved(width) = self.width {
            return width;
        } else {
            panic!("Unresolved width")
        }
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
enum BusSlice {
    Full,
    Index(u32),
    Range(u32, u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BusWidth {
    Resolved(u32),
    Unresolved,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BusConnectionType {
    Unspecified,
    Sequential,
    Combinational,
}

#[derive(Debug, Clone)]
enum WireConnection<'a> {
    Auxiliary,
    Local(Expression<'a>),
}

impl<'a> WireConnection<'a> {
    fn get_expression(&self) -> &Expression<'a> {
        if let WireConnection::Local(expression) = self {
            return expression;
        } else {
            panic!("Unexpected auxiliary connection");
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Expression<'a> {
    bitwise_reduction: Option<BitwiseOperator>,
    operators: Vec<BitwiseOperator>,
    operands: Vec<Operand<'a>>,
    width: BusWidth,
}

impl<'a> Expression<'a> {
    fn set_width(&mut self, width: u32) {
        match self.width {
            BusWidth::Unresolved => self.width = BusWidth::Resolved(width),
            BusWidth::Resolved(old_width) if old_width == width => (),
            BusWidth::Resolved(_) => panic!("Expression width mismatch"),
        };
    }
    fn from_pair(
        expression_pair: Pair<'a, Rule>,
        global_context: &GlobalContext,
        chip_context: &mut ChipContext<'a>,
    ) -> Expression<'a> {
        let mut expression = Expression {
            bitwise_reduction: None,
            operators: Vec::new(),
            operands: Vec::new(),
            width: BusWidth::Unresolved,
        };
        let mut expression_pairs = expression_pair.into_inner();
        let mut prev_token_is_operator = true;
        if let Some(expression_pair) = expression_pairs.peek() {
            if expression_pair.as_rule() == Rule::operator {
                expression.bitwise_reduction =
                    Some(BitwiseOperator::try_from(expression_pair.as_str()).unwrap());
                expression.set_width(1);
                expression_pairs.next();
            }
        }
        while let Some(expression_pair) = expression_pairs.next() {
            match expression_pair.as_rule() {
                Rule::operator => {
                    if prev_token_is_operator {
                        panic!("Two operators in a row in an expression")
                    }
                    prev_token_is_operator = true;
                    expression
                        .operators
                        .push(BitwiseOperator::try_from(expression_pair.as_str()).unwrap())
                }
                Rule::operand => {
                    if !prev_token_is_operator {
                        panic!("Two operands in a row in an expression")
                    }
                    prev_token_is_operator = false;
                    let operand_pair = expression_pair.into_inner().next().unwrap();
                    match operand_pair.as_rule() {
                        Rule::bits => {
                            let bits: Vec<bool> = operand_pair
                                .as_str()
                                .chars()
                                .map(|char| char == '1')
                                .collect();
                            expression.set_width(bits.len() as u32);
                            expression.operands.push(Operand::Bits(bits));
                        }
                        Rule::bus => {
                            let bus = Bus::from_pair(operand_pair);
                            if let BusSlice::Index(_) = bus.slice {
                                expression.set_width(1);
                            } else if let BusSlice::Range(start, end) = bus.slice {
                                expression.set_width(end - start);
                            }
                            chip_context
                                .bus_connections
                                .entry(bus.identifier)
                                .or_insert(BusConnection {
                                    input_connections: Vec::new(),
                                    output_connections: Vec::new(),
                                    bus_connection_type: BusConnectionType::Unspecified,
                                    width: BusWidth::Unresolved,
                                })
                                .output_connections
                                .push(bus.slice.clone());
                            expression.operands.push(Operand::Bus(bus));
                        }
                        Rule::conditional => {
                            expression
                                .operands
                                .push(Operand::Conditional(Conditional::from_pair(
                                    operand_pair,
                                    global_context,
                                    chip_context,
                                )));
                        }
                        Rule::expression => {
                            let operand_expression =
                                Expression::from_pair(operand_pair, global_context, chip_context);
                            if let BusWidth::Resolved(width) = operand_expression.width {
                                expression.set_width(width);
                            }
                            expression
                                .operands
                                .push(Operand::Expression(operand_expression));
                        }
                        Rule::array => {
                            expression
                                .operands
                                .push(Operand::Array(Array::from_pairs(operand_pair)));
                        }
                        Rule::chip_instantiation => expression.operands.push(
                            Operand::ChipInstantiation(ChipInstantiation::from_pair(
                                operand_pair,
                                global_context,
                                chip_context,
                            )),
                        ),
                        _ => unreachable!(),
                    }
                }

                _ => unreachable!(),
            }
        }
        expression
    }

    fn try_resolve_width(
        &self,
        global_context: &GlobalContext,
        chip_context: &ChipContext<'a>,
        explored_buses: &mut HashSet<&'a str>,
    ) -> Option<u32> {
        if let BusWidth::Resolved(width) = self.width {
            return Some(width);
        }
        for operand in self.operands.iter() {
            let width = operand.try_resolve_width(global_context, chip_context, explored_buses);
            if width.is_some() {
                return width;
            }
        }
        None
    }

    fn extend_netlist_for_index(
        &self,
        index: u32,
        chip_identifier: &str,
        root_name: &str,
        bus_nodes: &mut HashMap<&'a str, Vec<NodeIndex>>,
        global_context: &GlobalContext<'a>,
        chips: &HashMap<&str, ChipContext<'a>>,
        netlist: &mut Graph<String, NLConnection>,
        chip_instantiation_nodes: &mut Vec<(
            ChipInstantiation<'a>,
            HashMap<&'a str, Vec<NodeIndex>>,
        )>,
    ) -> NodeIndex {
        let chip_context = &chips[chip_identifier];
        if let Some(bitwise_operator) = &self.bitwise_reduction {
            if index != 0 {
                panic!("Invalid expression index");
            }
            let operand = &self.operands[0];
            let width = operand
                .try_resolve_width(global_context, chip_context, &mut HashSet::new())
                .unwrap();
            let root = if width > 1 {
                netlist.add_node(bitwise_operator.to_string())
            } else {
                operand.extend_netlist_for_index(
                    0,
                    chip_identifier,
                    root_name,
                    bus_nodes,
                    global_context,
                    chips,
                    netlist,
                    chip_instantiation_nodes,
                )
            };
            let mut current_node = root;
            for i in 0..width - 2 {
                let operand_node = operand.extend_netlist_for_index(
                    i,
                    chip_identifier,
                    root_name,
                    bus_nodes,
                    global_context,
                    chips,
                    netlist,
                    chip_instantiation_nodes,
                );
                let operator_node = netlist.add_node(bitwise_operator.to_string());
                netlist.add_edge(operand_node, current_node, NLConnection::Wire);
                netlist.add_edge(operator_node, current_node, NLConnection::Wire);
                current_node = operator_node;
            }
            if width > 1 {
                let left_operand_node = operand.extend_netlist_for_index(
                    width - 1,
                    chip_identifier,
                    root_name,
                    bus_nodes,
                    global_context,
                    chips,
                    netlist,
                    chip_instantiation_nodes,
                );
                let right_operand_node = operand.extend_netlist_for_index(
                    width - 2,
                    chip_identifier,
                    root_name,
                    bus_nodes,
                    global_context,
                    chips,
                    netlist,
                    chip_instantiation_nodes,
                );
                netlist.add_edge(right_operand_node, current_node, NLConnection::Wire);
                netlist.add_edge(left_operand_node, current_node, NLConnection::Wire);
            }
            return root;
        }
        let width = self
            .try_resolve_width(global_context, chip_context, &mut HashSet::new())
            .unwrap();
        if index >= width {
            if index != 0 {
                panic!(
                    "Invalid expression index: {} for width: {}, {:?}",
                    index, width, &self
                );
            }
        }
        let root = if self.operands.len() > 1 {
            netlist.add_node(self.operators[0].to_string())
        } else {
            self.operands[0].extend_netlist_for_index(
                index,
                chip_identifier,
                root_name,
                bus_nodes,
                global_context,
                chips,
                netlist,
                chip_instantiation_nodes,
            )
        };
        let mut current_node = root;
        for i in 1..self.operators.len() {
            let operand_node = self.operands[i - 1].extend_netlist_for_index(
                index,
                chip_identifier,
                root_name,
                bus_nodes,
                global_context,
                chips,
                netlist,
                chip_instantiation_nodes,
            );
            let operator_node = netlist.add_node(self.operators[i].to_string());
            netlist.add_edge(operand_node, current_node, NLConnection::Wire);
            netlist.add_edge(operator_node, current_node, NLConnection::Wire);
            current_node = operator_node;
        }
        if self.operands.len() > 1 {
            let left_operand_node = self.operands[self.operands.len() - 1]
                .extend_netlist_for_index(
                    index,
                    chip_identifier,
                    root_name,
                    bus_nodes,
                    global_context,
                    chips,
                    netlist,
                    chip_instantiation_nodes,
                );
            let right_operand_node = self.operands[self.operands.len() - 2]
                .extend_netlist_for_index(
                    index,
                    chip_identifier,
                    root_name,
                    bus_nodes,
                    global_context,
                    chips,
                    netlist,
                    chip_instantiation_nodes,
                );
            netlist.add_edge(right_operand_node, current_node, NLConnection::Wire);
            netlist.add_edge(left_operand_node, current_node, NLConnection::Wire);
        }
        root
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Operand<'a> {
    Conditional(Conditional<'a>),
    Expression(Expression<'a>),
    Array(Array<'a>),
    ChipInstantiation(ChipInstantiation<'a>),
    Bus(Bus<'a>),
    Bits(Vec<bool>),
}

impl<'a> Operand<'a> {
    fn try_resolve_width(
        &self,
        global_context: &GlobalContext,
        chip_context: &ChipContext<'a>,
        explored_buses: &mut HashSet<&'a str>,
    ) -> Option<u32> {
        match self {
            Operand::ChipInstantiation(chip_instantiation) => {
                Some(chip_instantiation.resolve_width(global_context))
            }
            Operand::Bits(bits) => Some(bits.len() as u32),
            Operand::Bus(bus) => {
                bus.try_resolve_width(global_context, chip_context, explored_buses)
            }
            Operand::Conditional(conditional) => {
                conditional.try_resolve_width(global_context, chip_context, explored_buses)
            }
            Operand::Expression(expression) => {
                expression.try_resolve_width(global_context, chip_context, explored_buses)
            }
            Operand::Array(array) => {
                array.try_resolve_width(global_context, chip_context, explored_buses)
            }
        }
    }

    fn extend_netlist_for_index(
        &self,
        index: u32,
        chip_identifier: &str,
        root_name: &str,
        bus_nodes: &mut HashMap<&'a str, Vec<NodeIndex>>,
        global_context: &GlobalContext<'a>,
        chips: &HashMap<&str, ChipContext<'a>>,
        netlist: &mut Graph<String, NLConnection>,
        chip_instantiation_nodes: &mut Vec<(
            ChipInstantiation<'a>,
            HashMap<&'a str, Vec<NodeIndex>>,
        )>,
    ) -> NodeIndex {
        match self {
            Operand::Expression(expression) => expression.extend_netlist_for_index(
                index,
                chip_identifier,
                root_name,
                bus_nodes,
                global_context,
                chips,
                netlist,
                chip_instantiation_nodes,
            ),
            Operand::Conditional(conditional) => conditional.extend_netlist_for_index(
                index,
                chip_identifier,
                root_name,
                bus_nodes,
                global_context,
                chips,
                netlist,
                chip_instantiation_nodes,
            ),
            Operand::Bits(bits) => netlist.add_node((bits[index as usize] as u8).to_string()),
            Operand::Bus(bus) => bus.extend_netlist_for_index(index, bus_nodes),
            Operand::Array(array) => array.extend_netlist_for_index(index, bus_nodes, netlist),
            Operand::ChipInstantiation(chip_instantiation) => chip_instantiation
                .extend_netlist_for_index(
                    index,
                    chip_identifier,
                    root_name,
                    bus_nodes,
                    global_context,
                    chips,
                    netlist,
                    chip_instantiation_nodes,
                ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChipInstantiation<'a> {
    chip_identifier: &'a str,
    input_connections: Vec<(Bus<'a>, BusConnectionType, Expression<'a>)>,
    output_index: Option<u32>,
}

impl<'a> ChipInstantiation<'a> {
    fn from_pair(
        pair: Pair<'a, Rule>,
        global_context: &GlobalContext,
        chip_context: &mut ChipContext<'a>,
    ) -> ChipInstantiation<'a> {
        let mut chip_instantiation_pairs = pair.into_inner();
        let identifier = chip_instantiation_pairs.next().unwrap().as_str();
        let assignments = chip_instantiation_pairs
            .map(|assignment_pair| parse_assignment(assignment_pair, global_context, chip_context))
            .collect();
        chip_context.requisite_chips.insert(identifier);
        ChipInstantiation {
            chip_identifier: identifier,
            input_connections: assignments,
            output_index: None,
        }
    }

    fn resolve_width(&self, global_context: &GlobalContext) -> u32 {
        let output_signatures = &global_context.chip_signatures[self.chip_identifier].1;
        output_signatures[self.output_index.unwrap_or(0) as usize].1
    }

    fn inner_logic_eq(&self, other: &Self) -> bool {
        self.chip_identifier == other.chip_identifier
            && self.input_connections == other.input_connections
    }

    fn extend_netlist_for_index(
        &self,
        index: u32,
        chip_identifier: &str,
        root_name: &str,
        bus_nodes: &mut HashMap<&'a str, Vec<NodeIndex>>,
        global_context: &GlobalContext<'a>,
        chips: &HashMap<&str, ChipContext<'a>>,
        netlist: &mut Graph<String, NLConnection>,
        chip_instantiation_nodes: &mut Vec<(
            ChipInstantiation<'a>,
            HashMap<&'a str, Vec<NodeIndex>>,
        )>,
    ) -> NodeIndex {
        for (precalculated_chip_instantiation, output_nodes) in chip_instantiation_nodes.iter() {
            if precalculated_chip_instantiation.inner_logic_eq(self) {
                let output_identifier = global_context.chip_signatures[self.chip_identifier].1
                    [self.output_index.unwrap_or(0) as usize]
                    .0;
                return output_nodes[output_identifier][index as usize];
            }
        }
        let mut chip_instantiation_input_nodes = HashMap::new();
        for (identifier, width) in global_context.chip_signatures[self.chip_identifier]
            .0
            .iter()
        {
            chip_instantiation_input_nodes.insert(
                *identifier,
                (0..*width)
                    .map(|index| {
                        netlist.add_node(format!(
                            "{}{}_{}{}",
                            root_name, self.chip_identifier, identifier, index
                        ))
                    })
                    .collect::<Vec<NodeIndex>>(),
            );
        }

        for (bus, bus_connection_type, expression) in self.input_connections.iter() {
            let edge_weight = match bus_connection_type {
                BusConnectionType::Combinational => NLConnection::Wire,
                BusConnectionType::Sequential => NLConnection::FlipFlop,
                BusConnectionType::Unspecified => panic!("Unspecified connection type"),
            };
            match bus.slice {
                BusSlice::Index(index) => {
                    let node = chip_instantiation_input_nodes[bus.identifier][index as usize];
                    let input_node = expression.extend_netlist_for_index(
                        0,
                        chip_identifier,
                        root_name,
                        bus_nodes,
                        global_context,
                        chips,
                        netlist,
                        chip_instantiation_nodes,
                    );
                    netlist.add_edge(input_node, node, edge_weight.clone());
                }
                BusSlice::Range(start, end) => {
                    for (expression_index, node) in (start..end)
                        .map(|index| chip_instantiation_input_nodes[bus.identifier][index as usize])
                        .enumerate()
                        .collect::<Vec<_>>()
                        .into_iter()
                    {
                        let input_node = expression.extend_netlist_for_index(
                            expression_index as u32,
                            chip_identifier,
                            root_name,
                            bus_nodes,
                            global_context,
                            chips,
                            netlist,
                            chip_instantiation_nodes,
                        );
                        netlist.add_edge(input_node, node, edge_weight.clone());
                    }
                }
                BusSlice::Full => {
                    for (expression_index, node) in (0..chip_instantiation_input_nodes
                        [bus.identifier]
                        .len())
                        .map(|index| chip_instantiation_input_nodes[bus.identifier][index as usize])
                        .enumerate()
                        .collect::<Vec<_>>()
                        .into_iter()
                    {
                        let input_node = expression.extend_netlist_for_index(
                            expression_index as u32,
                            chip_identifier,
                            root_name,
                            bus_nodes,
                            global_context,
                            chips,
                            netlist,
                            chip_instantiation_nodes,
                        );
                        netlist.add_edge(input_node, node, edge_weight.clone());
                    }
                }
            }
        }

        let output_nodes = extend_netlist_with_chip(
            self.chip_identifier,
            chip_instantiation_input_nodes,
            global_context,
            chips,
            netlist,
            &format!("{}{}-", root_name, chip_identifier),
        );
        let output_identifier = global_context.chip_signatures[self.chip_identifier].1
            [self.output_index.unwrap_or(0) as usize]
            .0;
        let output_node = output_nodes[output_identifier][index as usize];
        chip_instantiation_nodes.push((self.clone(), output_nodes));
        output_node
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ArrayElement<'a> {
    Bus(Bus<'a>),
    Bits(Vec<bool>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Array<'a>(Vec<ArrayElement<'a>>);

impl<'a> Array<'a> {
    fn from_pairs(pair: Pair<Rule>) -> Array {
        let mut array_elements = Vec::new();
        for array_element in pair.into_inner() {
            match array_element.as_rule() {
                Rule::bus => {
                    array_elements.push(ArrayElement::Bus(Bus::from_pair(array_element)));
                }
                Rule::bits => {
                    let bits: Vec<bool> = array_element
                        .as_str()
                        .chars()
                        .map(|char| char == '1')
                        .collect();
                    array_elements.push(ArrayElement::Bits(bits));
                }
                _ => unreachable!(),
            }
        }
        Array(array_elements)
    }

    fn try_resolve_width(
        &self,
        global_context: &GlobalContext,
        chip_context: &ChipContext<'a>,
        explored_buses: &mut HashSet<&'a str>,
    ) -> Option<u32> {
        self.0
            .iter()
            .map(|array_element| match array_element {
                ArrayElement::Bits(bits) => Some(bits.len() as u32),
                ArrayElement::Bus(bus) => {
                    bus.try_resolve_width(global_context, chip_context, explored_buses)
                }
            })
            .try_fold(0, |acc, width| width.map(|width| width + acc))
    }

    fn extend_netlist_for_index(
        &self,
        index: u32,
        bus_nodes: &mut HashMap<&'a str, Vec<NodeIndex>>,
        netlist: &mut Graph<String, NLConnection>,
    ) -> NodeIndex {
        let mut current_idx = 0;
        for array_element in self.0.iter() {
            match array_element {
                ArrayElement::Bits(bits) => {
                    if current_idx + bits.len() > index as usize {
                        return netlist
                            .add_node((bits[index as usize - current_idx] as u8).to_string());
                    } else {
                        current_idx += bits.len();
                    }
                }
                ArrayElement::Bus(bus) => {
                    let width = bus.get_width(bus_nodes) as usize;
                    if current_idx + width > index as usize {
                        return bus.extend_netlist_for_index(index - current_idx as u32, bus_nodes);
                    } else {
                        current_idx += width;
                    }
                }
            }
        }
        panic!("Index out of bounds")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Conditional<'a> {
    condition_path_pairs: Vec<(Expression<'a>, Expression<'a>)>,
    default: Expression<'a>,
}

impl<'a> Conditional<'a> {
    fn from_pair(
        pair: Pair<'a, Rule>,
        global_context: &GlobalContext,
        chip_context: &mut ChipContext<'a>,
    ) -> Conditional<'a> {
        let mut expressions = pair
            .into_inner()
            .map(|pair| Expression::from_pair(pair, global_context, chip_context));
        let mut condition_path_pairs = Vec::new();
        let mut default = None;
        while let Some(expression) = expressions.next() {
            if let Some(second_expression) = expressions.next() {
                condition_path_pairs.push((expression, second_expression));
            } else {
                default = Some(expression);
            }
        }
        Conditional {
            condition_path_pairs,
            default: default.unwrap(),
        }
    }

    fn try_resolve_width(
        &self,
        global_context: &GlobalContext,
        chip_context: &ChipContext<'a>,
        explored_buses: &mut HashSet<&'a str>,
    ) -> Option<u32> {
        self.condition_path_pairs
            .iter()
            .find_map(|(_, path)| {
                path.try_resolve_width(global_context, chip_context, explored_buses)
            })
            .or(self
                .default
                .try_resolve_width(global_context, chip_context, explored_buses))
    }

    fn extend_netlist_for_index(
        &self,
        index: u32,
        chip_identifier: &str,
        root_name: &str,
        bus_nodes: &mut HashMap<&'a str, Vec<NodeIndex>>,
        global_context: &GlobalContext<'a>,
        chips: &HashMap<&str, ChipContext<'a>>,
        netlist: &mut Graph<String, NLConnection>,
        chip_instantiation_nodes: &mut Vec<(
            ChipInstantiation<'a>,
            HashMap<&'a str, Vec<NodeIndex>>,
        )>,
    ) -> NodeIndex {
        let root = netlist.add_node(BitwiseOperator::Or.to_string());
        let mut current_or_node = root;
        for current_pair_index in 0..self.condition_path_pairs.len() {
            let root_and_node = netlist.add_node(BitwiseOperator::And.to_string());
            let mut current_and_node = root_and_node;
            for negated_condition_index in 0..current_pair_index {
                let condition_node = self.condition_path_pairs[negated_condition_index]
                    .0
                    .extend_netlist_for_index(
                        0,
                        chip_identifier,
                        root_name,
                        bus_nodes,
                        global_context,
                        chips,
                        netlist,
                        chip_instantiation_nodes,
                    );
                let negation_node = netlist.add_node(BitwiseOperator::Not.to_string());
                let next_and_node = netlist.add_node(BitwiseOperator::And.to_string());
                netlist.add_edge(condition_node, negation_node, NLConnection::Wire);
                netlist.add_edge(negation_node, current_and_node, NLConnection::Wire);
                netlist.add_edge(next_and_node, current_and_node, NLConnection::Wire);
                current_and_node = next_and_node;
            }
            let condition_node = self.condition_path_pairs[current_pair_index]
                .0
                .extend_netlist_for_index(
                    0,
                    chip_identifier,
                    root_name,
                    bus_nodes,
                    global_context,
                    chips,
                    netlist,
                    chip_instantiation_nodes,
                );
            let path_node = self.condition_path_pairs[current_pair_index]
                .1
                .extend_netlist_for_index(
                    index,
                    chip_identifier,
                    root_name,
                    bus_nodes,
                    global_context,
                    chips,
                    netlist,
                    chip_instantiation_nodes,
                );
            netlist.add_edge(condition_node, current_and_node, NLConnection::Wire);
            netlist.add_edge(path_node, current_and_node, NLConnection::Wire);
            netlist.add_edge(root_and_node, current_or_node, NLConnection::Wire);
            if current_pair_index != self.condition_path_pairs.len() - 1 {
                let next_or_node = netlist.add_node(BitwiseOperator::Or.to_string());
                netlist.add_edge(next_or_node, current_or_node, NLConnection::Wire);
                current_or_node = next_or_node;
            }
        }
        let root_and_node = netlist.add_node(BitwiseOperator::And.to_string());
        let mut current_and_node = root_and_node;
        for negated_condition_index in 0..self.condition_path_pairs.len() {
            let condition_node = self.condition_path_pairs[negated_condition_index]
                .0
                .extend_netlist_for_index(
                    0,
                    chip_identifier,
                    root_name,
                    bus_nodes,
                    global_context,
                    chips,
                    netlist,
                    chip_instantiation_nodes,
                );

            let negation_node = netlist.add_node(BitwiseOperator::Not.to_string());
            netlist.add_edge(condition_node, negation_node, NLConnection::Wire);
            netlist.add_edge(negation_node, current_and_node, NLConnection::Wire);
            if negated_condition_index != self.condition_path_pairs.len() - 1 {
                let next_and_node = netlist.add_node(BitwiseOperator::And.to_string());
                netlist.add_edge(next_and_node, current_and_node, NLConnection::Wire);
                current_and_node = next_and_node;
            }
        }
        let default_path_node = self.default.extend_netlist_for_index(
            index,
            chip_identifier,
            root_name,
            bus_nodes,
            global_context,
            chips,
            netlist,
            chip_instantiation_nodes,
        );
        netlist.add_edge(default_path_node, current_and_node, NLConnection::Wire);
        netlist.add_edge(root_and_node, current_or_node, NLConnection::Wire);
        root
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Bus<'a> {
    identifier: &'a str,
    slice: BusSlice,
}

impl<'a> Bus<'a> {
    fn from_pair(bus_pair: Pair<Rule>) -> Bus {
        {
            let mut bus_pairs = bus_pair.into_inner();
            let bus_identifier = bus_pairs.next().unwrap().as_str();
            let bus_slice = match bus_pairs.next() {
                None => BusSlice::Full,
                Some(bus_pair) => match bus_pair.as_rule() {
                    Rule::range => {
                        let mut range_pairs = bus_pair.into_inner();
                        BusSlice::Range(
                            range_pairs.next().unwrap().as_str().parse().unwrap(),
                            range_pairs.next().unwrap().as_str().parse().unwrap(),
                        )
                    }
                    Rule::index => BusSlice::Index(
                        bus_pair
                            .into_inner()
                            .next()
                            .unwrap()
                            .as_str()
                            .parse()
                            .unwrap(),
                    ),
                    _ => unreachable!(),
                },
            };
            Bus {
                identifier: bus_identifier,
                slice: bus_slice,
            }
        }
    }
    fn try_resolve_width(
        &self,
        global_context: &GlobalContext,
        chip_context: &ChipContext<'a>,
        explored_buses: &mut HashSet<&'a str>,
    ) -> Option<u32> {
        if explored_buses.contains(self.identifier) {
            return None;
        } else {
            explored_buses.insert(self.identifier);
        }
        match self.slice {
            BusSlice::Index(_) => Some(1),
            BusSlice::Range(start, end) => Some(end - start),
            BusSlice::Full => chip_context.bus_connections[self.identifier].try_resolve_width(
                global_context,
                chip_context,
                explored_buses,
            ),
        }
    }
    fn get_width(&self, bus_nodes: &HashMap<&'a str, Vec<NodeIndex>>) -> u32 {
        match self.slice {
            BusSlice::Index(_) => 1,
            BusSlice::Range(start, end) => end - start,
            BusSlice::Full => bus_nodes[self.identifier].len() as u32,
        }
    }
    fn extend_netlist_for_index(
        &self,
        index: u32,
        bus_nodes: &mut HashMap<&'a str, Vec<NodeIndex>>,
    ) -> NodeIndex {
        match self.slice {
            BusSlice::Full => bus_nodes[self.identifier][index as usize],
            BusSlice::Index(bus_index) => {
                if index != 0 {
                    panic!("Non-zero index {} used on an indexed bus {:?}", index, self)
                } else {
                    bus_nodes[self.identifier][bus_index as usize]
                }
            }
            BusSlice::Range(start, end) => {
                if start + index >= end {
                    panic!("Index {} used on a sliced bus {:?}", index, self)
                } else {
                    bus_nodes[self.identifier][(index + start) as usize]
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BitwiseOperator {
    And,
    Or,
    Xor,
    Not,
    Nand,
    Nor,
    Xnor,
}

impl fmt::Display for BitwiseOperator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BitwiseOperator::And => write!(f, "and"),
            BitwiseOperator::Or => write!(f, "or"),
            BitwiseOperator::Xor => write!(f, "xor"),
            BitwiseOperator::Not => write!(f, "not"),
            BitwiseOperator::Nand => write!(f, "nand"),
            BitwiseOperator::Nor => write!(f, "nor"),
            BitwiseOperator::Xnor => write!(f, "xnor"),
        }
    }
}

impl TryFrom<&str> for BitwiseOperator {
    type Error = &'static str;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "and" | "&" => Ok(BitwiseOperator::And),
            "nand" | "!&" => Ok(BitwiseOperator::Nand),
            "or" | "|" => Ok(BitwiseOperator::Or),
            "nor" | "!|" => Ok(BitwiseOperator::Nor),
            "xor" | "^" | "!=" => Ok(BitwiseOperator::Xor),
            "xnor" | "!^" | "==" => Ok(BitwiseOperator::Xnor),
            "not" | "!" => Ok(BitwiseOperator::Not),
            _ => Err("String must be one of 'and' , 'or' , 'xor' , 'not' , 'nand' , 'nor' , 'xnor' , '&' , ',' , '^' , '!' , '!&' , '!,' , '!^' , '==' , '!='"),
        }
    }
}

fn parse_assignment<'a>(
    assignment_pair: Pair<'a, Rule>,
    global_context: &GlobalContext,
    chip_context: &mut ChipContext<'a>,
) -> (Bus<'a>, BusConnectionType, Expression<'a>) {
    let mut assignment_pairs = assignment_pair.into_inner();
    let lhs = Bus::from_pair(assignment_pairs.next().unwrap());
    let assignment_type = match assignment_pairs.next().unwrap().as_rule() {
        Rule::combinational_assignment => BusConnectionType::Combinational,
        Rule::sequential_assignment => BusConnectionType::Sequential,
        _ => unreachable!(),
    };
    let rhs = Expression::from_pair(
        assignment_pairs.next().unwrap(),
        global_context,
        chip_context,
    );
    (lhs, assignment_type, rhs)
}

fn get_chip_signature(chip: Pair<Rule>) -> (&str, (Vec<(&str, u32)>, Vec<(&str, u32)>)) {
    let mut pairs = chip.into_inner();
    let identifier = pairs.next().unwrap().as_str();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for pair in pairs {
        match pair.as_rule() {
            Rule::input => {
                let mut input_pairs = pair.into_inner();
                inputs.push((
                    input_pairs.next().unwrap().as_str(),
                    input_pairs
                        .next()
                        .map(|width| width.into_inner().next().unwrap().as_str().parse().unwrap())
                        .unwrap_or(1),
                ));
            }
            Rule::output => {
                let mut output_pairs = pair.into_inner();
                outputs.push((
                    output_pairs.next().unwrap().as_str(),
                    output_pairs
                        .next()
                        .map(|width| width.into_inner().next().unwrap().as_str().parse().unwrap())
                        .unwrap_or(1),
                ));
            }
            _ => break,
        }
    }
    (identifier, (inputs, outputs))
}

fn _format_pair(pair: Pair<Rule>, indent_level: usize, is_newline: bool) -> String {
    let indent = if is_newline {
        "  ".repeat(indent_level)
    } else {
        String::new()
    };

    let children: Vec<_> = pair.clone().into_inner().collect();
    let len = children.len();
    let children: Vec<_> = children
        .into_iter()
        .map(|pair| {
            _format_pair(
                pair,
                if len > 1 {
                    indent_level + 1
                } else {
                    indent_level
                },
                len > 1,
            )
        })
        .collect();

    let dash = if is_newline { "- " } else { "" };
    let pair_tag = match pair.as_node_tag() {
        Some(tag) => format!("(#{}) ", tag),
        None => String::new(),
    };
    match len {
        0 => format!(
            "{}{}{}{}: {:?}",
            indent,
            dash,
            pair_tag,
            format!("{:?}", pair.as_rule()),
            pair.as_span().as_str()
        ),
        1 => format!(
            "{}{}{}{} > {}",
            indent,
            dash,
            pair_tag,
            format!("{:?}", pair.as_rule()),
            children[0]
        ),
        _ => format!(
            "{}{}{}{}\n{}",
            indent,
            dash,
            pair_tag,
            format!("{:?}", pair.as_rule()),
            children.join("\n")
        ),
    }
}
