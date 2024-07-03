use std::fs;
mod parse;

fn main() {
    let hdl_text = fs::read_to_string("input.rhdl").expect("No HDL?");
    let _result = parse::parse_hdl(&hdl_text);
}
