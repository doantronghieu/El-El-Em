import type { PageLoad } from './$types';
import { api, getErrorMessage } from '$api';

export const load = (async ({ params }) => {
	try {
		const {
			data: { pdf, download_url }
		} = await api.get(`/pdfs/${params.id}`);

		return {
			document: pdf,
			documentUrl: download_url
		};
	} catch (err) {
		return {
			error: getErrorMessage(err)
		};
	}
}) satisfies PageLoad;
