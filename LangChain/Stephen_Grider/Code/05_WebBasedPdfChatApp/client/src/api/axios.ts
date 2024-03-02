import axios from 'axios';
import { addError } from '$s/errors';

interface ApiError {
	message: string;
	error: string;
}

export const api = axios.create({
	baseURL: '/api'
});

api.interceptors.response.use(
	(res) => res,
	(err) => {
		if (err.response && err.response.status >= 500) {
			const { response } = err;
			const message = getErrorMessage(err);

			if (message) {
				addError({
					contentType: response.headers['Content-Type'] || response.headers['content-type'],
					message: getErrorMessage(err)
				});
			}
		}
		return Promise.reject(err);
	}
);

export const getErrorMessage = (error: unknown) => {
	if (axios.isAxiosError(error)) {
		const apiError = error.response?.data as ApiError;
		if (typeof apiError === 'string' && (apiError as string).length > 0) {
			return apiError;
		}
		return apiError?.message || apiError?.error || error.message;
	}

	if (error instanceof Error) {
		return error.message;
	}

	if (
		error &&
		typeof error === 'object' &&
		'message' in error &&
		typeof error.message === 'string'
	) {
		return error.message;
	}

	return 'Something went wrong';
};

export const getError = (error: unknown) => {
	if (axios.isAxiosError(error)) {
		const apiError = error.response?.data as ApiError;
		return apiError;
	}

	return null;
};
