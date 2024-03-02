import type { Document } from '$s/documents';
import type { PageLoad } from './$types';
import { api, getErrorMessage } from '$api';

export const load = (async () => {
	try {
		const { data } = await api.get<Document[]>('/pdfs');
		return {
			documents: data
		};
	} catch (err) {
		return {
			error: getErrorMessage(err)
		};
	}
}) satisfies PageLoad;
