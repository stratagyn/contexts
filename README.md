# contexts

A `ContextManager` encapsulates a collection of `HashMap<K, V, S>` and treats them as a
singular map. The first map in the collection is considered the primary or *local* context.

Searching starts with the local context and proceeds until a value is found or there are no
more maps to check. This behavior can be affected by specifying an index to start from, or
limiting searching to the local context only.

Insertions and removals are to and from the local context. Managers initialized via
`ContextManager::new` or `ContextManager::with_capacity` do not start with an initial context.
The `ContextManager::with_empty` and `ContextManager::from` methods create managers with one or
more initial contexts. Inserts and removes have no effect until a first context is pushed.

Managers can be cloned from any point in the underlying collection.

Context managers do not currently support direct iteration over key-value pairs, however
any manager can be *collapsed* into a single `HashMap` or `BTreeMap` and iterated from
there. Keys in these maps will have their most recently associated value from the manager.

## Example

```rust
use std::collections::HashMap;
use contexts::ContextManager;

fn main() {
    let mut manager = ContextManager::with_empty();
    
    manager.insert("red", 255u8); //[{"red":255}]
    
    if manager.contains_key("red") { 
        println!("red in context") 
    } else { 
        println!("red not in context") 
    }

    match manager.get("green") {
        Some(_) => println!("green in context"),
        None => println!("green not in context") 
    }

    manager.push(HashMap::from([("red", 63u8)])); //[{"red":63}, {"red":255}]

    manager.push_empty(); //[{}, {"red":63}, {"red":255}]
    
    println!("red = {}", &manager["red"]);

    match manager.get_from(1, "red") {
        Some(byte) => println!("non-local red = {}", byte),
        None => println!("no value set for red in non-local contexts")
    }
    
    match manager.get_local("red") {
        Some(byte) => println!("locally red = {}", byte),
        None => println!("no value set for red in local context")
    }
    
    manager.pop(); //[{"red":63}, {"red":255}]

    println!("after pop red = {}", &manager["red"]);

    match manager.get_from(1, "red") {
        Some(byte) => println!("after pop non-local red = {}", byte),
        None => println!("after pop no value set for red in non-local contexts")
    }

    match manager.get_local("red") {
        Some(byte) => println!("after pop locally red = {}", byte),
        None => println!("after pop no value set for red in local context")
    }
    
    manager.push_local(); //[{"red":63}, {"red":63}, {"red":255}]
    
    if let Some(b) = manager.get_mut("red") {
        *b = 192u8; //[{"red":192}, {"red":63}, {"red":255}]
    }

    println!("after mut red = {}", &manager["red"]);

    manager.remove("red"); //[{}, {"red":63}, {"red":255}]

    println!("after remove red = {}", &manager["red"]);

    match manager.get_local("red") {
        Some(byte) => println!("after remove locally red = {}", byte),
        None => println!("after remove no value set for red in local context")
    }
    
    let fork = manager.fork().unwrap(); //[{}]
    let fork2 = manager.fork_from(1).unwrap(); //[{}, {"red":63}]

    println!("# of contexts in manager = {}", manager.len());
    println!("# of contexts in fork = {}", fork.len());
    println!("# of contexts in second fork = {}", fork.len());
    
    manager.remove_all("red"); //[{}, {}, {}]

    match manager.get("red") {
        Some(byte) => println!("after remove all red = {}", byte),
        None => println!("after remove all no value set for red")
    }
}
```

Prints:

```
red in context
green not in context
red = 63
non-local red = 63
no value set for red in local context
after pop red = 63
after pop non-local red = 255
after pop locally red = 63
after mut red = 192
after remove red = 63
after remove no value set for red in local context
# of contexts in manager = 3
# of contexts in fork = 1
# of contexts in second fork = 2
after remove all no value set for red
```