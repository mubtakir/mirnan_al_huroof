# English ADL in al_aql

English syntax is supported beside Arabic syntax so `al_aql` can describe programming plans, software architecture, code behavior, and technical domains naturally.

## Core Forms

```adl
class Service < thing { }
class Database < thing { }

api : Service { latency: 0.3 }
db : Database { latency: 0.6 }

verb query { target: load, scale: 1.0 }
relation api calls db
quantifier all Service has endpoint
comparison db greater than api in latency
intent { actor: api, action: query, target: db, intent: fetch_user, goal: return_profile, actual_result: rows_loaded }
exception { rule: cache_read, condition: cache.enabled == true, exception: cache_miss, priority: 5 }
metaphor { expression: data flows through pipeline, source_domain: river, target_domain: software, literal_subject: data, borrowed_actor: river, action: flows, transferred_property: movement }
```

## Conditional Templates

Blocks can be written across multiple lines:

```adl
class Material < thing { }
water : Material { boiling_point: 100.0 }
salt : Material { quantity: 1.0 }
verb raise_boiling { target: boiling_point, scale: 2.0 }

template conditional salt_raises_boiling {
  domain: chemistry,
  source: Material,
  action: add,
  target: Material,
  condition: source.quantity > 0,
  result_actor: target,
  result_action: raise_boiling,
  result_state: boiling_point_increased,
  confidence: 0.9
}
```

When this is inferred:

```julia
infer_event!(space, "salt", "add", "water")
```

the system records the causal frame and applies `raise_boiling` to `water`.

## Why This Matters For Code Planning

English ADL can describe code as causal and relational knowledge:

```text
Service calls Database
Cache miss triggers Database query
Function validates input to prevent invalid state
Component depends on interface
```

This lets Mirnan organize programming ideas as:

```text
entities + relations + actions + conditions + intended goals + actual results
```

which is the same structure used for general understanding.
