import { writable } from 'svelte/store';

const count = writable(0);

export { count };
