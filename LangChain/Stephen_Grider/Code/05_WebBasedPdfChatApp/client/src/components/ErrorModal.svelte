<script lang="ts">
	import { onMount } from 'svelte';
	import { errorStore, reset } from '$s/errors';
	import ErrorMessage from '$c/ErrorMessage.svelte';

	const listener = (event: KeyboardEvent) => {
		if (event.key === 'Escape') {
			reset();
		}
	};

	onMount(() => {
		window.addEventListener('keydown', listener);
		return () => window.removeEventListener('keydown', listener);
	});
</script>

{#if $errorStore.errors.length}
	<div on:click={reset} on:keypress={reset} class="fixed z-10 inset-0 bg-gray-500 opacity-40" />
	<div class="fixed modal z-20 top-0 right-0 bottom-0 w-6/12 bg-red-300 p-3 flex flex-col gap-1">
		<div class="absolute top-2.5 right-2.5">
			<button on:click={reset}>Close</button>
		</div>
		<h1 class="text-3xl text-white-500 font-bold">An Error Occured...</h1>
		<div class="flex-1 bg-white overflow-y-scroll">
			{#each $errorStore.errors as error}
				{#if error.contentType?.includes('text/html')}
					<ErrorMessage message={error.message} />
				{:else}
					<p class="p-2 text-red-500">{error.message}</p>
				{/if}
			{/each}
		</div>
	</div>
{/if}
