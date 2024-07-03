# Rat HDL
A Hardware Decription Language for Rats!

This repo is a proof of concept for a custom HDL compiler. It takes a single `input.rhdl` file containing chips written in Rat HDL and outputs a single DOT graph representing the whole circuit in boolean logic.
Sequential and combinational parts of the circuit are in the same graph, but the edges are annotated to allow for easy splitting into combinational subgraphs.
Rat HDL does type-checking! If you try to plug a 6 bit slice of a 4 bit bus into a 2 bit output you *will* be yelled at.

This project is *very* proof of concept. It's some of the jankiest Rust I've ever written as I was learning concepts on the go and was mostly interested in gauging the difficulty level of writing such a thing rather than having an actually good implementation.

This means there are semantic errors in the language definition itself, type-checking works wherever I felt like implementing it and isn't very smart. But it does work!
An example rhdl input:
```
chip ExampleChip (a<6>, b<6>) -> (out) {
    silly_and = [a[0..2], b[1], 011] and b; //bitwise and on 6-bit buses
    somevar = if &(a == b) { a[2] xor a[0] } else { b[1] }; //if else assignment, must have same type
    out = xor [silly_and, somevar]; //xor reduce all bits
}

chip ComplexChip(large_bus<10>) -> (out, last_two<2>) {
    intermediate_result = ExampleChip(a[0]=large_bus[8], a[1..6] = large_bus[3..8], b=large_bus[4..10]);
    final_result = if large_bus[9] {
        intermediate_result
    } else {
        chungus
    };
    chungus <= final_result;
    out=final_result;
    last_two = large_bus[8..10] and large_bus[6..8];
}

chip Top(in<10>) -> (out) {
    intermediate_out, last_two = ComplexChip(large_bus=in);
    out = intermediate_out and (|last_two); //and followed by or reduction
}
```
Gets turned into a graph *correctly*!
![graph](https://github.com/Miav771/rhdl/assets/31471893/67fd3d11-0e6a-4d66-b5c1-0c0319e958d6)

This project will be revisited with a better implementation at some point, but right now I'm trying to gain some more insight on the next steps in the EDA pipeline before optimizing prematurely!
