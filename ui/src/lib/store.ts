import { useSyncExternalStore } from "react";

/**
 * Minimal external store built on `useSyncExternalStore` — avoids a state
 * library and context re-render storms. Components subscribe to a slice via a
 * selector and only re-render when that slice changes by reference/value.
 */
export interface Store<T> {
  get: () => T;
  set: (next: Partial<T> | ((s: T) => T)) => void;
  subscribe: (cb: () => void) => () => void;
  use: <S>(selector: (s: T) => S) => S;
}

export function createStore<T extends object>(initial: T): Store<T> {
  let state = initial;
  const subs = new Set<() => void>();

  const get = () => state;
  const set: Store<T>["set"] = (next) => {
    state =
      typeof next === "function"
        ? (next as (s: T) => T)(state)
        : { ...state, ...next };
    subs.forEach((s) => s());
  };
  const subscribe = (cb: () => void) => {
    subs.add(cb);
    return () => {
      subs.delete(cb);
    };
  };
  const use = <S,>(selector: (s: T) => S): S =>
    useSyncExternalStore(
      subscribe,
      () => selector(state),
      () => selector(state),
    );

  return { get, set, subscribe, use };
}
