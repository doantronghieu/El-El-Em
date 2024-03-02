<script lang="ts">
	export let label: string = '';
</script>

<div>
	<div class="flex justify-between items-center">
		<label for="" class="block text-sm mb-2">{label}</label>
	</div>
	<div class="relative">
		<slot />
	</div>
</div>
