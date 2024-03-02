// import axios from 'axios';
import { writable } from 'svelte/store';
import { api, getErrorMessage } from '$api';

export interface Document {
	id: string;
	file_id: string;
	name: string;
}

interface UploadStore {
	data: Document[];
	error: string;
	uploadProgress: number;
}

const INITIAL_STATE = {
	data: [],
	error: '',
	uploadProgress: 0
};

const documents = writable<UploadStore>(INITIAL_STATE);

const set = (val: Partial<UploadStore>) => {
	documents.update((state) => ({ ...state, ...val }));
};

const setUploadProgress = (event: ProgressEvent) => {
	const progress = Math.round((event.loaded / event.total) * 100);

	set({ uploadProgress: progress });
};

const upload = async (file: File) => {
	set({ error: '' });

	try {
		const formData = new FormData();
		formData.append('file', file);

		await api.post('/pdfs', formData, {
			onUploadProgress: setUploadProgress
		});
	} catch (error) {
		return set({ error: getErrorMessage(error) });
	}
};

const getDocuments = async () => {
	const { data } = await api.get('/pdfs');
	set({ data });
};

const clearErrors = () => {
	set({ error: '', uploadProgress: 0 });
};

export { upload, getDocuments, documents, clearErrors };
