//! A [ContextManager] encapsulates a collection of
//! [HashMap<K, V, S>](https://doc.rust-lang.org/std/collections/struct.HashMap.html) and treats
//! them as a singular map. The first map in the collection is considered the primary or *local*
//! context.
//!
//! Searching starts with the local context and proceeds until a value is found or there are no
//! more maps to check. This behavior can be affected by specifying an index to start from, or
//! limiting searching to the local context only.
//!
//! Insertions and removals are to and from the local context. Managers initialized via
//! [ContextManager::new] or [ContextManager::with_capacity] do not start with an initial context.
//! The [ContextManager::with_empty] and [ContextManager::from] methods create managers with one or
//! more initial contexts. Inserts and removes have no effect until a first context is pushed.
//!
//! Managers can be cloned from any point in the underlying collection.
//!
//! Context managers do not currently support direct iteration over key-value pairs, however
//! any manager can be *collapsed* into a single `HashMap` or `BTreeMap` and iterated from
//! there. Keys in these maps will have their most recently associated value from the manager.
//!
//! ## Examples
//!
//! ```rust
//! use std::collections::HashMap;
//! use contexts::ContextManager;
//!
//! let mut manager = ContextManager::with_empty();
//!
//! manager.insert("red", 255u8); //[{"red":255}]
//!
//! if manager.contains_key("red") {
//!     println!("red in context")
//! } else {
//!     println!("red not in context")
//! }
//!
//! match manager.get("green") {
//!     Some(_) => println!("green in context"),
//!     None => println!("green not in context")
//! }
//!
//! manager.push(HashMap::from([("red", 63u8)])); //[{"red":63}, {"red":255}]
//! manager.push_empty(); //[{}, {"red":63}, {"red":255}]
//!
//! println!("red = {}", &manager["red"]);
//!
//! match manager.get_from(1, "red") {
//!     Some(byte) => println!("non-local red = {}", byte),
//!     None => println!("no value set for red in non-local contexts")
//! }
//!
//! match manager.get_local("red") {
//!     Some(byte) => println!("locally red = {}", byte),
//!     None => println!("no value set for red in local context")
//! }
//!
//! manager.pop(); //[{"red":63}, {"red":255}]
//!
//! println!("after pop red = {}", &manager["red"]);
//!
//! match manager.get_from(1, "red") {
//!     Some(byte) => println!("after pop non-local red = {}", byte),
//!     None => println!("after pop no value set for red in non-local contexts")
//! }
//!
//! match manager.get_local("red") {
//!     Some(byte) => println!("after pop locally red = {}", byte),
//!     None => println!("after pop no value set for red in local context")
//! }
//!
//! manager.push_local(); //[{"red":63}, {"red":63}, {"red":255}]
//! if let Some(b) = manager.get_mut("red") {
//!     *b = 192u8; //[{"red":192}, {"red":63}, {"red":255}]
//! }
//!
//! println!("after mut red = {}", &manager["red"]);
//!
//! manager.remove("red"); //[{}, {"red":63}, {"red":255}]
//!
//! println!("after remove red = {}", &manager["red"]);
//!
//! match manager.get_local("red") {
//!     Some(byte) => println!("after remove locally red = {}", byte),
//!     None => println!("after remove no value set for red in local context")
//! }
//!
//! let fork = manager.fork().unwrap(); //[{}]
//! let fork2 = manager.fork_from(1).unwrap(); //[{}, {"red":63}]
//!
//! println!("# of contexts in manager = {}", manager.len());
//! println!("# of contexts in fork = {}", fork.len());
//! println!("# of contexts in second fork = {}", fork.len());
//!
//! manager.remove_all("red"); //[{}, {}, {}]
//!
//! match manager.get("red") {
//!     Some(byte) => println!("after remove all red = {}", byte),
//!     None => println!("after remove all no value set for red")
//! }
//! ```
//!
//! Prints:
//!
//! ```plaintext
//! red in context
//! green not in context
//! red = 63
//! non-local red = 63
//! no value set for red in local context
//! after pop red = 63
//! after pop non-local red = 255
//! after pop locally red = 63
//! after mut red = 192
//! after remove red = 63
//! after remove no value set for red in local context
//! # of contexts in manager = 3
//! # of contexts in fork = 1
//! # of contexts in second fork = 2
//! after remove all no value set for red
//! ```

use std::borrow::Borrow;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{BuildHasher, Hash, RandomState};
use std::ops::Index;

/// A singular view into a collection of `HashMap<K, V, S>`, each referred to as a context.
#[derive(Debug)]
pub struct ContextManager<K, V, S = RandomState> {
    inner: VecDeque<HashMap<K, V, S>>
}


impl<K, V> ContextManager<K, V, RandomState>
where K: Hash + Eq {
    /// Creates a context manager initialized with an empty context.
    ///
    /// # Example
    /// ```
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::with_empty();
    ///
    /// manager.insert("x", 1);
    ///
    /// assert_eq!(&manager["x"], &1);
    /// ```
    pub fn with_empty() -> Self { Self { inner: VecDeque::from([HashMap::new()]) } }

    /// Creates an empty context manager.
    ///
    /// # Example
    /// ```
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::new();
    ///
    /// manager.insert("x", 1);
    ///
    /// assert!(!manager.contains_key("x"));
    /// ```
    pub fn new() -> Self { Self::default() }

    /// Creates an empty context manager with space for `capacity` contexts.
    ///
    /// # Example
    /// ```
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::with_capacity(3);
    ///
    /// manager.insert("x", 1);
    ///
    /// assert!(!manager.contains_key("x"));
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self { inner: VecDeque::with_capacity(capacity) }
    }

    /// Aggregates all contexts into a single map where keys have their most recent value.
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::with_capacity(3);
    ///
    /// manager.push(HashMap::from([("y", 3)]));
    /// manager.push(HashMap::from([("w", 1), ("x", 2)]));
    /// manager.push(HashMap::from([("y", 4), ("z", 3)]));
    ///
    /// let map = manager.collapse();
    ///
    /// assert_eq!(&map["w"], &1);
    /// assert_eq!(&map["x"], &2);
    /// assert_eq!(&map["y"], &4);
    /// assert_eq!(&map["z"], &3);
    /// ```
    pub fn collapse(mut self) -> HashMap<K, V> {
        if self.inner.len() == 1 {
            self.inner.pop_front().unwrap()
        } else {
            let mut map = HashMap::new();

            loop {
                if self.inner.is_empty() { break; }

                let next = self.inner.pop_back().unwrap();

                map.extend(next);
            }

            map
        }
    }

    /// Adds an empty local context.
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::with_empty();
    ///
    /// manager.insert("x", 1);
    ///
    /// assert_eq!(manager.get_local("x"), Some(&1));
    ///
    /// manager.push_empty();
    ///
    /// assert_eq!(manager.get_local("x"), None);
    /// ```
    pub fn push_empty(&mut self) { self.inner.push_front(HashMap::new()) }
}


impl<K, V, S> ContextManager<K, V, S> {
    /// Returns the number of contexts in the manager.
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::<&str, i32>::new();
    ///
    /// assert!(manager.is_empty());
    ///
    /// manager.push_empty();
    ///
    /// assert!(!manager.is_empty());
    ///
    /// manager = ContextManager::with_empty();
    ///
    /// assert!(!manager.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool { self.inner.is_empty() }

    /// Returns the number of contexts in the manager.
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::<&str, i32>::new();
    ///
    /// assert_eq!(manager.len(), 0);
    ///
    /// manager.push_empty();
    ///
    /// assert_eq!(manager.len(), 1);
    ///
    /// manager = ContextManager::with_empty();
    ///
    /// assert_eq!(manager.len(), 1);
    /// ```
    pub fn len(&self) -> usize { self.inner.len() }

    /// Removes the local context if one is present.
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::with_empty();
    ///
    /// manager.insert("x", 1);
    ///
    /// let popped = manager.pop();
    ///
    /// assert!(popped.is_some());
    /// assert_eq!(popped.unwrap().get("x"), Some(&1));
    /// assert_eq!(manager.pop(), None);
    /// ```
    pub fn pop(&mut self) -> Option<HashMap<K, V, S>> { self.inner.pop_front() }

    /// Adds a new local context.
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::with_empty();
    ///
    /// manager.insert("x", 1);
    ///
    /// assert_eq!(&manager["x"], &1);
    /// assert_eq!(manager.get("y"), None);
    ///
    /// manager.push(HashMap::from([("y", 2)]));
    ///
    /// assert_eq!(&manager["y"], &2);
    /// ```
    pub fn push(&mut self, context: HashMap<K, V, S>) { self.inner.push_front(context) ; }
}


impl<K, V, S> ContextManager<K, V, S>
where K: Hash + Eq, S: BuildHasher {
    /// Aggregates all contexts storing each key and its most recent value into `src`.
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::with_capacity(3);
    ///
    /// manager.push(HashMap::from([("y", 4)]));
    /// manager.push(HashMap::from([("w", 2), ("x", 3)]));
    ///
    /// let mut map = HashMap::from([("v", 1), ("x", 2), ("z", 5)]);
    ///
    /// manager.collapse_into(&mut map);
    ///
    /// assert_eq!(&map["v"], &1);
    /// assert_eq!(&map["w"], &2);
    /// assert_eq!(&map["x"], &3);
    /// assert_eq!(&map["y"], &4);
    /// assert_eq!(&map["z"], &5);
    /// ```
    pub fn collapse_into(mut self, src: &mut HashMap<K, V, S>) {
        loop {
            if self.inner.is_empty() { break; }

            let next = self.inner.pop_back().unwrap();

            src.extend(next);
        }
    }

    /// Whether a key is present in the context.
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::with_capacity(3);
    ///
    /// manager.push(HashMap::from([("w", 1)]));
    /// manager.push(HashMap::from([("x", 2)]));
    ///
    /// assert!(manager.contains_key(&"w"));
    /// assert!(manager.contains_key(&"x"));
    /// assert!(!manager.contains_key(&"y"));
    /// ```
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where K: Borrow<Q>, Q: ?Sized + Hash + Eq {
        for map in &self.inner {
            if map.contains_key(key) {
                return true;
            }
        }

        false
    }

    /// Whether a key is present in the local context
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::new();
    ///
    /// manager.push(HashMap::from([("w", 1)]));
    ///
    /// assert!(manager.contains_local_key("w"));
    ///
    /// manager.push_empty();
    ///
    /// assert!(!manager.contains_local_key("w"));
    /// ```
    pub fn contains_local_key<Q>(&self, key: &Q) -> bool
    where K: Borrow<Q>, Q: ?Sized + Hash + Eq {
        self.inner.len() > 0 && self.inner[0].contains_key(key)
    }

    /// Returns a reference to the value associated with `key`.
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::with_capacity(3);
    ///
    /// manager.push(HashMap::from([("w", 1)]));
    /// manager.push(HashMap::from([("x", 2)]));
    ///
    /// assert_eq!(manager.get(&"w"), Some(&1));
    /// assert_eq!(manager.get(&"x"), Some(&2));
    /// assert_eq!(manager.get(&"y"), None);
    /// ```
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where K: Borrow<Q>, Q: ?Sized + Hash + Eq {
        self.inner.iter().find_map(|ctx| ctx.get(key))
    }

    /// Returns a vector of references to all values associated with `key`, ordered by
    /// precedence.
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::with_capacity(3);
    ///
    /// manager.push(HashMap::from([("w", 1)]));
    /// manager.push(HashMap::from([("w", 2)]));
    /// manager.push(HashMap::from([("w", 3)]));
    ///
    /// assert_eq!(manager.get_all(&"w"), vec![&3, &2, &1]);
    /// assert_eq!(manager.get_all(&"x"), Vec::<&i32>::new());
    /// ```
    pub fn get_all<Q>(&self, key: &Q) -> Vec<&V>
    where K: Borrow<Q>, Q: ?Sized + Hash + Eq {
        self.inner.iter().filter_map(|map| map.get(key)).collect()
    }

    /// Returns a reference to the value associated with `key` starting with the context at `index`.
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::with_capacity(3);
    ///
    /// manager.push(HashMap::from([("w", 1)]));
    /// manager.push(HashMap::from([("x", 2)]));
    /// manager.push(HashMap::from([("w", 3)]));
    ///
    /// assert_eq!(manager.get_from(0, &"w"), Some(&3));
    /// assert_eq!(manager.get_from(1, &"w"), Some(&1));
    /// assert_eq!(manager.get_from(2, &"w"), Some(&1));
    /// assert_eq!(manager.get_from(3, &"w"), None);
    /// ```
    pub fn get_from<Q>(&self, index: usize, key: &Q) -> Option<&V>
    where K: Borrow<Q>, Q: ?Sized + Hash + Eq {
        self.inner.range(index..).find_map(|ctx| ctx.get(key))
    }

    /// Returns a reference to the value associated with `key` in the local context.
    ///
    /// # Example
    ///```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::from([("w", 1)]);
    ///
    /// assert_eq!(manager.get_local(&"w"), Some(&1));
    ///
    /// manager.push_empty();
    ///
    /// assert_eq!(manager.get_local(&"w"), None);
    /// ```
    pub fn get_local<Q>(&self, key: &Q) -> Option<&V>
    where K: Borrow<Q>, Q: ?Sized + Hash + Eq {
        if self.inner.is_empty() { None } else { self.inner[0].get(key) }
    }

    /// Returns a mutable reference to the value associated with `key` in the local context.
    ///
    /// # Example
    ///```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::from([("w", 1)]);
    ///
    /// if let Some(w) = manager.get_local_mut("w") {
    ///     *w = 2;
    /// }
    ///
    /// assert_eq!(manager.get(&"w"), Some(&2));
    ///
    /// manager.push_empty();
    ///
    /// assert_eq!(manager.get_local_mut(&"w"), None);
    /// ```
    pub fn get_local_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where K: Borrow<Q>, Q: ?Sized + Hash + Eq {
        if self.inner.is_empty() { None } else { self.inner[0].get_mut(key) }
    }

    /// Returns a mutable reference to the value associated with `key`.
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::from([("w", 1)]);
    ///
    /// if let Some(v) = manager.get_mut("w") {
    ///     *v = 2;
    /// }
    ///
    /// assert_eq!(&manager["w"], &2);
    /// assert_eq!(manager.get_mut(&"x"), None);
    /// ```
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where K: Borrow<Q>, Q: ?Sized + Hash + Eq {
        self.inner.iter_mut().find_map(|ctx| ctx.get_mut(key))
    }

    /// Returns a reference to the value associated with `key` starting with the context at `index`.
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::new();
    ///
    /// manager.push(HashMap::from([("w", 1)]));
    /// manager.push(HashMap::from([("w", 2)]));
    ///
    /// if let Some(v) = manager.get_mut_from(1, "w") {
    ///     *v = 3;
    /// }
    ///
    /// assert_eq!(manager.get_from(0, "w"), Some(&2));
    /// assert_eq!(manager.get_from(1, "w"), Some(&3));
    /// assert_eq!(manager.get_mut_from(2, "w"), None);
    /// ```
    pub fn get_mut_from<Q>(&mut self, index: usize, key: &Q) -> Option<&mut V>
    where K: Borrow<Q>, Q: ?Sized + Hash + Eq {
        self.inner.range_mut(index..).find_map(|ctx| ctx.get_mut(key))
    }

    /// Associates `value` with `key` in the local context if there is one.
    ///
    /// # Example
    /// ```
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::new();
    ///
    /// assert_eq!(manager.insert("w", 1), None);
    /// assert_eq!(manager.get("w"), None);
    ///
    /// manager.push_empty();
    ///
    /// manager.insert("w", 1);
    ///
    /// assert_eq!(manager.insert("w", 2), Some(1));
    ///
    /// manager.push_empty();
    ///
    /// assert_eq!(manager.insert("w", 3), None);
    /// ```
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.inner.is_empty() { None } else { self.inner[0].insert(key, value) }
    }

    /// Removes `key` from the local context if one is present.
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::with_capacity(3);
    ///
    /// assert_eq!(manager.remove("w"), None);
    ///
    /// manager.push(HashMap::from([("w", 1)]));
    /// manager.push(HashMap::from([("w", 2)]));
    ///
    /// assert_eq!(manager.remove("w"), Some(2));
    /// assert_eq!(manager.remove("w"), None);
    /// assert_eq!(&manager["w"], &1);
    /// ```
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where K: Borrow<Q>, Q: ?Sized + Hash + Eq {
        if self.inner.is_empty() { None } else { self.inner[0].remove(key) }
    }

    /// Removes all instances of `key` from the context manager, returning a vector of the values,
    /// ordered by precedence.
    ///
    /// # Examples
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::with_capacity(3);
    ///
    /// manager.push(HashMap::from([("w", 1)]));
    /// manager.push(HashMap::from([("w", 2)]));
    /// manager.push(HashMap::from([("w", 3)]));
    ///
    /// assert_eq!(manager.remove_all(&"w"), vec![3, 2, 1]);
    /// assert_eq!(manager.remove_all(&"x"), vec![]);
    /// ```
    pub fn remove_all<Q>(&mut self, key: &Q) -> Vec<V>
    where K: Borrow<Q>, Q: ?Sized + Hash + Eq {
        self.inner.iter_mut().filter_map(|ctx| ctx.remove(key)).collect()
    }
}


impl<K, V, S> ContextManager<K, V, S>
where K: Hash + Eq + Clone, V: Clone, S: BuildHasher + Clone {
    /// Creates a new context manager initialized with a clone of the current local context.
    ///
    /// Equivalent to `manager.fork_from(0)`
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::with_capacity(3);
    ///
    /// assert_eq!(manager.fork(), None);
    ///
    /// manager.push(HashMap::from([("w", 1)]));
    /// manager.push(HashMap::from([("x", 2)]));
    /// manager.push(HashMap::from([("y", 3)]));
    ///
    /// let forked = manager.fork().unwrap();
    ///
    /// assert_eq!(forked.get("w"), None);
    /// assert_eq!(forked.get("x"), None);
    /// assert_eq!(&forked["y"], &3);
    /// ```
    pub fn fork(&self) -> Option<ContextManager<K, V, S>> {
        if self.inner.is_empty() { None } else { Some(ContextManager::from(self.inner[0].clone())) }
    }

    /// Creates a new context manager initialized with clones of all contexts from the local one up
    /// to and including the one at `index`.
    ///
    /// `manager.fork_from(manager.len() - 1)` is equivalent to `manager.clone()`.
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::with_capacity(3);
    ///
    /// manager.push(HashMap::from([("w", 1)]));
    /// manager.push(HashMap::from([("x", 2)]));
    /// manager.push(HashMap::from([("y", 3)]));
    ///
    /// let forked = manager.fork_from(1).unwrap();
    ///
    /// assert_eq!(forked.get("w"), None);
    /// assert_eq!(&forked["x"], &2);
    /// assert_eq!(&forked["y"], &3);
    ///
    /// let invalid_fork = manager.fork_from(3);
    ///
    /// assert_eq!(invalid_fork, None);
    /// ```
    pub fn fork_from(&self, index: usize) -> Option<ContextManager<K, V, S>> {
        if index >= self.inner.len() {
            None
        } else {
            Some(ContextManager {
                inner: self.inner
                    .range(0..(index + 1)).map(|ctx| ctx.clone())
                    .collect()
            })
        }
    }

    /// Adds a new context that is a clone of the local context, if one is present.
    ///
    /// # Example
    /// ```
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::from([("w", 1)]);
    /// manager.push_local();
    ///
    /// assert_eq!(&manager["w"], &1);
    ///
    /// let w = manager.get_mut("w").unwrap();
    ///
    /// *w = 2;
    ///
    /// assert_eq!(&manager["w"], &2);
    ///
    /// manager.pop();
    ///
    /// assert_eq!(&manager["w"], &1);
    /// ```
    pub fn push_local(&mut self) {
        if self.inner.len() > 0 {
            let context = self.inner[0].clone();

            self.inner.push_front(context);
        }
    }

    /// Adds a new local context merged with the previous local context.
    ///
    /// The new context has higher precedence.
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::from([("w", 1)]);
    /// let context = HashMap::from([("x", 2)]);
    ///
    /// manager.push_with_local(context);
    ///
    /// assert_eq!(manager.get_local("w"), Some(&1));
    /// assert_eq!(&manager["x"], &2);
    ///
    /// manager.pop();
    ///
    /// assert_eq!(&manager["w"], &1);
    /// assert_eq!(manager.get("x"), None);
    /// ```
    pub fn push_with_local(&mut self, context: HashMap<K, V, S>) {
        if self.inner.is_empty() {
            self.inner.push_back(context)
        } else {
            let mut plocal = self.inner[0].clone();

            plocal.extend(context);

            self.inner.push_front(plocal);
        }
    }
}

impl<K, V, S> ContextManager<K, V, S> 
where K: Ord {
    /// Aggregates all contexts into a single map where keys have their most recent value.
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::with_capacity(3);
    ///
    /// manager.push(HashMap::from([("y", 3)]));
    /// manager.push(HashMap::from([("w", 1), ("x", 2)]));
    /// manager.push(HashMap::from([("y", 4), ("z", 3)]));
    ///
    /// let map = manager.collapse_ordered();
    ///
    /// assert_eq!(&map["w"], &1);
    /// assert_eq!(&map["x"], &2);
    /// assert_eq!(&map["y"], &4);
    /// assert_eq!(&map["z"], &3);
    /// ```
    pub fn collapse_ordered(mut self) -> BTreeMap<K, V> {
        let mut map = BTreeMap::new();

        loop {
            if self.inner.is_empty() { break; }

            let next = self.inner.pop_back().unwrap();

            map.extend(next);
        }

        map
    }

    /// Aggregates all contexts storing each key and its most recent value into `src`.
    ///
    /// # Example
    /// ```
    /// # use std::collections::{BTreeMap, HashMap};
    /// # use contexts::ContextManager;
    /// let mut manager = ContextManager::with_capacity(3);
    ///
    /// manager.push(HashMap::from([("y", 4)]));
    /// manager.push(HashMap::from([("w", 2), ("x", 3)]));
    ///
    /// let mut map = BTreeMap::from([("v", 1), ("x", 2), ("z", 5)]);
    ///
    /// manager.collapse_into_ordered(&mut map);
    ///
    /// assert_eq!(&map["v"], &1);
    /// assert_eq!(&map["w"], &2);
    /// assert_eq!(&map["x"], &3);
    /// assert_eq!(&map["y"], &4);
    /// assert_eq!(&map["z"], &5);
    /// ```
    pub fn collapse_into_ordered(mut self, src: &mut BTreeMap<K, V>) {
        loop {
            if self.inner.is_empty() { break; }

            let next = self.inner.pop_back().unwrap();

            src.extend(next);
        }
    }
}
impl<K, V, S> Clone for ContextManager<K, V, S>
where K: Clone, V: Clone, S: Clone{
    fn clone(&self) -> Self { Self { inner: self.inner.clone() } }
}


impl<K, V, S> Default for ContextManager<K, V, S> {
    /// Creates an empty `ContextManager<K, V, S>`
    fn default() -> Self { Self { inner: VecDeque::new() } }
}


impl <K, V, S> Extend<(K, V)> for ContextManager<K, V, S>
where K: Hash + Eq, S: BuildHasher + Default {
    /// Adds key-value pairs from an iterator to the context manager.
    ///
    /// If the context manager is empty, a new `HashMap<K, V, S>` is created with the default
    /// hasher.
    fn extend<I: IntoIterator<Item=(K, V)>>(&mut self, iter: I) {
        if self.inner.is_empty() {
            self.inner.push_front(HashMap::from_iter(iter));
        } else {
            self.inner[0].extend(iter);
        }
    }
}


impl<K, V, S> From<HashMap<K, V, S>> for ContextManager<K, V, S>
where K: Hash + Eq, S: BuildHasher {
    /// Creates a new `ContextManager<K, V, S>` with `initial` as the first context.
    fn from(initial: HashMap<K, V, S>) -> Self {
        Self { inner: VecDeque::from([initial]) }
    }
}


impl<K, V, const N: usize> From<[(K, V); N]> for ContextManager<K, V, RandomState>
where K: Hash + Eq {
    /// Creates a new `ContextManager<K, V>` with a first context initialized from the key-value
    /// pairs in `initial`.
    ///
    /// Repeated keys will have all but one of the values dropped.
    fn from(initial: [(K, V); N]) -> Self {
        Self { inner: VecDeque::from([HashMap::from(initial)]) }
    }
}


impl<K, V, S, const N: usize> From<[HashMap<K, V, S>; N]> for ContextManager<K, V, S>
where K: Hash + Eq, S: BuildHasher {
    /// Creates a new `ContextManager<K, V>` initialized with the contexts in `initial`.
    ///
    /// Precedence proceeds from the first context toward the last.
    fn from(initial: [HashMap<K, V, S>; N]) -> Self {
        Self { inner: VecDeque::from(initial) }
    }
}


impl<K, V, S> FromIterator<(K, V)> for ContextManager<K, V, S>
where K: Hash + Eq, S: BuildHasher + Default {
    /// Creates a new `ContextManager<K, V>` with a first context initialized from the key-value
    /// pairs in `initial`.
    ///
    /// Repeated keys will have all but one of the values dropped.
    fn from_iter<I: IntoIterator<Item=(K, V)>>(initial: I) -> Self {
        Self { inner: VecDeque::from([HashMap::from_iter(initial)]) }
    }
}


impl<K, V, S> FromIterator<HashMap<K, V, S>> for ContextManager<K, V, S>
where K: Hash + Eq, S: BuildHasher {
    /// Creates a new `ContextManager<K, V>` initialized with the contexts in `initial`.
    ///
    /// Precedence proceeds from the first context toward the last.
    fn from_iter<I: IntoIterator<Item=HashMap<K, V, S>>>(iter: I) -> Self {
        Self { inner: VecDeque::from_iter(iter) }
    }
}


impl<K, Q, V, S> Index<&Q> for ContextManager<K, V, S>
where K: Hash + Eq + Borrow<Q>, Q: ?Sized + Hash + Eq, S: BuildHasher {
    type Output = V;

    /// Returns a reference to the value associated with `key`.
    ///
    /// Panics if the context manager is empty, or `key` is not found in any contexts.
    fn index(&self, key: &Q) -> &V {
        self.inner.iter().find_map(|ctx| ctx.get(key)).expect("key not found")
    }
}


impl<K, V, S> PartialEq for ContextManager<K, V, S>
where K: Hash + Eq, V: PartialEq, S: BuildHasher {
    fn eq(&self, other: &ContextManager<K, V, S>) -> bool {
        self.inner.eq(&other.inner)
    }
}


impl<K, V, S> Eq for ContextManager<K, V, S>
where K: Hash + Eq, V: Eq, S: BuildHasher {}