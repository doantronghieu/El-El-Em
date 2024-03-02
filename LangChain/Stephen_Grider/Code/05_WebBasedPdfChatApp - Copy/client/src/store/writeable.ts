import { writable as base } from 'svelte/store';
import produce, { enableMapSet } from 'immer';

enableMapSet();

export interface Writable<T> {
	set(this: void, value: T): void;
	update(this: void, updater: (value: T) => void): void;
	subscribe(this: void, run: (value: T) => void): () => void;
	get(): T;
}

export const writable = <T>(value: T): Writable<T> => {
	const store = base(value);

	let val = value;
	store.subscribe((v) => {
		val = v;
	});

	return {
		...store,
		get: () => {
			return val;
		},
		update: (fn: (value: T) => void) => {
			store.update((value) => {
				return produce(value, fn);
			});
		}
	};
};
