import { writable } from 'svelte/store';

interface ApiError {
	message: string;
	contentType?: string;
}

interface ErrorStore {
	errors: ApiError[];
}

const INITIAL_STATE = {
	errors: []
};

const errorStore = writable<ErrorStore>(INITIAL_STATE);

const addError = (error: ApiError) => {
	errorStore.update((state) => {
		return { errors: [...state.errors, error] };
	});
};

const removeError = (error: ApiError) => {
	errorStore.update((state) => {
		return {
			errors: state.errors.filter((e) => e !== error)
		};
	});
};

const reset = () => {
	errorStore.set({ errors: [] });
};

export { addError, reset, removeError, errorStore };
