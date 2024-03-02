<script lang="ts">
	import { goto, beforeNavigate } from '$app/navigation';
	import Alert from '$c/Alert.svelte';
	import Button from '$c/Button.svelte';
	import { documents, upload, clearErrors } from '$s/documents';
	import Progress from '$c/Progress.svelte';

	let files: FileList;
	let loading = false;
	let uploadComplete = false;

	async function handleSubmit() {
		loading = true;
		await upload(files[0]);
		if (!$documents.error) {
			uploadComplete = true;

			setTimeout(() => {
				goto('/documents');
				loading = false;
			}, 2000);
		} else {
			loading = false;
		}
	}

	beforeNavigate(clearErrors);
</script>

<div class="w-full max-w-md mx-auto p-6">
	<h2 class="text-3xl font-bold m-10">Upload a Document</h2>
	<form on:submit|preventDefault={handleSubmit}>
		<div class="w-42">
			<label for="file-input" class="sr-only">Choose file</label>
			<input
				bind:files
				type="file"
				name="file-input"
				id="file-input"
				class="block w-full border border-gray-200 shadow-sm rounded-md text-sm focus:z-10 focus:border-blue-500 focus:ring-blue-500 dark:bg-slate-900 dark:border-gray-700 dark:text-gray-400
        file:mr-4 file:py-2 file:px-4 file:m-4
        file:rounded-full file:border-0
        file:text-sm file:font-semibold
        file:bg-blue-50 file:text-violet-700
        hover:file:bg-violet-100"
			/>
		</div>

		<div class="my-4" />

		{#if loading && !$documents.error}
			<Progress progress={$documents.uploadProgress}>
				<Alert type="success">Upload Complete! Returning to list...</Alert>
			</Progress>
		{/if}

		{#if $documents.error}
			<Alert>Error: {$documents.error}</Alert>
		{/if}

		{#if !loading}
			<Button className="w-full my-3" disabled={loading}>Submit</Button>
		{/if}
	</form>
</div>
