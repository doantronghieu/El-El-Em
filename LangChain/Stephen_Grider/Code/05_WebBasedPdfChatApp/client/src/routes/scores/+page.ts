import type { PageLoad } from './$types';
import type { Scores } from '$s/scores';
import { api, getErrorMessage } from '$api';

export const load = (async () => {
	try {
		const { data } = await api.get<Scores>('/scores');

		return {
			scores: data
		};
	} catch (err) {
		return {
			error: getErrorMessage(err)
		};
	}
}) satisfies PageLoad;
